import numpy as np
import torch
import torch.nn.functional as F
import math

def swap_vecs(hidden,pos,head_id,vec,head_dim,args):
    assert len(pos)==len(vec), 'no_eq_len_condition option is now deprecated'
    assert len(hidden.shape)==3
    new_hidden = hidden.clone()
    if args.multihead:
        new_hidden[:,:,head_dim*head_id:head_dim*(head_id+1)][:,pos,:] = vec.to(args.device).clone()
    else:
        new_hidden[head_id,:,head_dim*head_id:head_dim*(head_id+1)][pos,:] = vec.to(args.device).clone()
    return new_hidden

def extract_attn_layer(layer_id,model,args):
    if args.model.startswith('bert'):
        layer = model.bert.encoder.layer[layer_id].attention.self
    elif args.model.startswith('roberta'):
        layer = model.roberta.encoder.layer[layer_id].attention.self
    elif args.model.startswith('albert'):
        layer = model.albert.encoder.albert_layer_groups[0].albert_layers[0].attention
    else:
        raise NotImplementedError("invalid model name")
    return layer

@torch.no_grad()
def layer_intervention(layer_id,layer,interventions,hidden,args):
    if args.model.startswith('bert') or args.model.startswith('roberta'):
        attention_layer = layer.attention.self
    elif args.model.startswith('albert'):
        attention_layer = layer.attention
    num_heads = attention_layer.num_attention_heads
    head_dim = attention_layer.attention_head_size

    qry = attention_layer.query(hidden)
    key = attention_layer.key(hidden)
    val = attention_layer.value(hidden)

    # swap representations
    for head_id in range(num_heads):
        for rep_type in ['layer','query','key','value']:
            if f'{rep_type}_{layer_id}_{head_id}' in interventions:
                for (pos,vec) in interventions[f'{rep_type}_{layer_id}_{head_id}']:
                    if rep_type=='layer':
                        hidden = swap_vecs(hidden,pos,head_id,vec,head_dim,args)
                    elif rep_type=='query':
                        qry = swap_vecs(qry,pos,head_id,vec,head_dim,args)
                    elif rep_type=='key':
                        key = swap_vecs(key,pos,head_id,vec,head_dim,args)
                    elif rep_type=='value':
                        val = swap_vecs(val,pos,head_id,vec,head_dim,args)

    #split into multiple heads
    split_qry = qry.view(*(qry.size()[:-1]+(num_heads,head_dim))).permute(0,2,1,3)
    split_key = key.view(*(key.size()[:-1]+(num_heads,head_dim))).permute(0,2,1,3)
    split_val = val.view(*(val.size()[:-1]+(num_heads,head_dim))).permute(0,2,1,3)

    #calculate the attention matrix
    attn_mat = F.softmax(split_qry@split_key.permute(0,1,3,2)/math.sqrt(head_dim),dim=-1)

    for head_id in range(num_heads):
        if f'attention_{layer_id}_{head_id}' in interventions:
            assert len(attn_mat.shape)==4
            if args.multihead:
                attn_mat[:,head_id,:,:] = interventions[f'attention_{layer_id}_{head_id}'].clone()
            else:
                assert attn_mat.shape[0]==num_heads
                attn_mat[head_id,head_id,:,:] = interventions[f'attention_{layer_id}_{head_id}'].clone()

    z_rep_indiv = attn_mat@split_val
    z_rep = z_rep_indiv.permute(0,2,1,3).reshape(*hidden.size())
    for head_id in range(num_heads):
        if f'z_rep_{layer_id}_{head_id}' in interventions:
            for (pos,vec) in interventions[f'z_rep_{layer_id}_{head_id}']:
                z_rep = swap_vecs(z_rep,pos,head_id,vec,head_dim,args)

    if args.model.startswith('bert') or args.model.startswith('roberta'):
        hidden_post_attn_res = layer.attention.output.dense(z_rep)+hidden # residual connection
        hidden_post_attn = layer.attention.output.LayerNorm(hidden_post_attn_res) # layer_norm

        hidden_post_interm = layer.intermediate(hidden_post_attn) # massive feed forward
        hidden_post_interm_res = layer.output.dense(hidden_post_interm)+hidden_post_attn # residual connection
        new_hidden =  layer.output.LayerNorm(hidden_post_interm_res) # layer_norm
    elif args.model.startswith('albert'):
        from transformers.modeling_utils import apply_chunking_to_forward
        hidden_post_attn_res = layer.attention.dense(z_rep)+hidden
        hidden_post_attn = layer.attention.LayerNorm(hidden_post_attn_res)

        ffn_output = apply_chunking_to_forward(layer.ff_chunk,layer.chunk_size_feed_forward,
                                                layer.seq_len_dim,hidden_post_attn)
        new_hidden = layer.full_layer_layer_norm(ffn_output+hidden_post_attn)
    return (new_hidden, attn_mat, [split_qry,split_key,split_val], z_rep)

def skeleton_model(start_layer_id,start_hidden,model,interventions,args):
    torch.set_num_threads(4)
    if args.model.startswith('bert'):
        core_model = model.bert
        lm_head = model.cls
    elif args.model.startswith('roberta'):
        core_model = model.roberta
        lm_head = model.lm_head
    elif args.model.startswith('albert'):
        core_model = model.albert
        lm_head = model.predictions
    else:
        raise NotImplementedError("invalid model name")
    output_hidden = []
    output_attn_mat = []
    output_qry = []
    output_key = []
    output_val = []
    output_z_rep = []
    hidden = start_hidden.clone()
    output_hidden.append(hidden)
    with torch.no_grad():
        for layer_id in range(start_layer_id, model.config.num_hidden_layers):
            if args.model.startswith('albert'):
                layer = model.albert.encoder.albert_layer_groups[0].albert_layers[0]
            else:
                layer = core_model.encoder.layer[layer_id]
            (hidden, attn_mat, qkv, z_rep) = layer_intervention(layer_id,layer,interventions,hidden,args)
            output_hidden.append(hidden)
            output_attn_mat.append(attn_mat)
            output_qry.append(qkv[0].to('cpu'))
            output_key.append(qkv[1].to('cpu'))
            output_val.append(qkv[2].to('cpu'))
            output_z_rep.append(z_rep.to('cpu'))
        logits = lm_head(hidden)
    return (logits,output_hidden,output_attn_mat,[output_qry,output_key,output_val],output_z_rep)
