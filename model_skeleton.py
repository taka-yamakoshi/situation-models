import numpy as np
import torch
import torch.nn.functional as F
import math

def swap_vecs(hidden,pos,vec,args):
    if len(pos)==len(vec):
        hidden[:,pos,:] = vec.clone()
        new_hidden = hidden.clone()
    else:
        new_hidden = torch.cat([hidden[0][:pos[0]],
                                vec.clone(),
                                hidden[0][(pos[-1]+1):]]).unsqueeze(0).to(args.device)
    return new_hidden

def swap_vecs_test(input_sent,pos,tokens,args):
    if len(pos)==len(tokens):
        input_sent[:,pos] = tokens.clone()
        new_input_sent = input_sent.clone()
    else:
        new_input_sent = torch.cat([input_sent[0][:pos[0]],
                                    tokens.clone(),
                                    input_sent[0][(pos[-1]+1):]]).unsqueeze(0).to(args.device)
    return new_input_sent


@torch.no_grad()
def layer_intervention(layer_id,layer,interventions,hidden,args):
    num_heads = layer.attention.self.num_attention_heads
    head_dim = layer.attention.self.attention_head_size

    # if the intervention is layer only, apply the intervention first
    if f'layer_{layer_id}' in interventions and f'query_{layer_id}' not in interventions:
        for (pos,vec) in interventions[f'layer_{layer_id}']:
            hidden = swap_vecs(hidden,pos,vec,args)
        key = layer.attention.self.key(hidden)
        query = layer.attention.self.query(hidden)
        value = layer.attention.self.value(hidden)

    # otherwise, calculate key, query, and value for attention first
    else:
        key = layer.attention.self.key(hidden)
        query = layer.attention.self.query(hidden)
        value = layer.attention.self.value(hidden)

        # swap representations
        if f'layer_{layer_id}' in interventions:
            for (pos,vec) in interventions[f'layer_{layer_id}']:
                hidden = swap_vecs(hidden,pos,vec,args)
        #NOTE: this swaps representations for all heads:
        #when doing intervenstions on each head, engineer the intervenstions accordingly
        if f'key_{layer_id}' in interventions:
            for (pos,vec) in interventions[f'key_{layer_id}']:
                key = swap_vecs(key,pos,vec,args)
        if f'query_{layer_id}' in interventions:
            for (pos,vec) in interventions[f'query_{layer_id}']:
                query = swap_vecs(query,pos,vec,args)
        if f'value_{layer_id}' in interventions:
            for (pos,vec) in interventions[f'value_{layer_id}']:
                value = swap_vecs(value,pos,vec,args)

    #split into multiple heads
    split_key = key.view(*(key.size()[:-1]+(num_heads,head_dim))).permute(0,2,1,3)
    split_query = query.view(*(query.size()[:-1]+(num_heads,head_dim))).permute(0,2,1,3)
    split_value = value.view(*(value.size()[:-1]+(num_heads,head_dim))).permute(0,2,1,3)

    #calculate the attention matrix
    attn_mat = F.softmax(split_query@split_key.permute(0,1,3,2)/math.sqrt(head_dim),dim=-1)

    z_rep_indiv = attn_mat@split_value
    z_rep = z_rep_indiv.permute(0,2,1,3).reshape(*hidden.size())

    hidden_post_attn_res = layer.attention.output.dense(z_rep)+hidden # residual connection
    hidden_post_attn = layer.attention.output.LayerNorm(hidden_post_attn_res) # layer_norm

    hidden_post_interm = layer.intermediate(hidden_post_attn) # massive feed forward
    hidden_post_interm_res = layer.output.dense(hidden_post_interm)+hidden_post_attn # residual connection
    return layer.output.LayerNorm(hidden_post_interm_res) # layer_norm

def skeleton_model(start_layer_id,start_hidden,model,model_name,interventions,args):
    if model_name.startswith('bert'):
        core_model = model.bert
        lm_head = model.cls
    elif model_name.startswith('roberta'):
        core_model = model.roberta
        lm_head = model.lm_head
    hidden = start_hidden.clone()
    with torch.no_grad():
        for layer_id,layer in zip(np.arange(start_layer_id,len(core_model.encoder.layer)),core_model.encoder.layer[start_layer_id:]):
            hidden = layer_intervention(layer_id,layer,interventions,hidden,args)
        logits = lm_head(hidden)
    return logits
