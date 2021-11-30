import numpy as np
import torch
import torch.nn.functional as F
import math

def swap_vecs(hidden,pos,vec,args):
    if len(pos)==len(vec):
        new_hidden = hidden.clone()
        new_hidden[:,pos,:] = vec.clone()
    else:
        new_hidden = torch.cat([hidden[0][:pos[0]].clone(),
                                vec.clone(),
                                hidden[0][(pos[-1]+1):].clone()]).unsqueeze(0).to(args.device)
    return new_hidden

def args_setup_deberta(args,init_shape):
    args.attention_mask = torch.ones(init_shape).to(args.device)
    if args.attention_mask.dim() <= 2:
        args.input_mask = args.attention_mask
    else:
        args.input_mask = (args.attention_mask.sum(-2) > 0).byte()
    args.attention_mask = args.encoder.get_attention_mask(args.attention_mask).to(args.device)
    return args

def args_setup_albert(args,init_shape):
    attention_mask = torch.ones(init_shape).to(args.device)
    args.extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    args.extended_attention_mask = args.extended_attention_mask.to(dtype=args.dtype)  # fp16 compatibility
    args.extended_attention_mask = (1.0-args.extended_attention_mask) * -10000.0
    return args

@torch.no_grad()
def layer_intervention(layer_id,layer,interventions,hidden,args):
    if args.model.startswith('deberta'):
        if f'layer_{layer_id}' in interventions:
            for (pos,vec) in interventions[f'layer_{layer_id}']:
                hidden = swap_vecs(hidden,pos,vec,args)
            args = args_setup_deberta(args, hidden.shape[:-1])
        if layer_id==0:
            init_hidden = hidden.clone()
        hidden = layer(hidden,attention_mask=args.attention_mask,rel_embeddings=args.rel_embeddings)
        if layer_id==0:
            hidden = args.encoder.conv(init_hidden, hidden, args.input_mask)
        return hidden
    elif args.model.startswith('albert'):
        if f'layer_{layer_id}' in interventions:
            for (pos,vec) in interventions[f'layer_{layer_id}']:
                hidden = swap_vecs(hidden,pos,vec,args)
            args = args_setup_albert(args, hidden.shape[:-1])
        layers_per_group = int(args.encoder.config.num_hidden_layers/args.encoder.config.num_hidden_groups)
        group_idx = int(layer_id/(args.encoder.config.num_hidden_layers/args.encoder.config.num_hidden_groups))
        hidden = args.encoder.albert_layer_groups[group_idx](hidden,
                                                            args.extended_attention_mask,
                                                            args.head_mask[group_idx*layers_per_group:(group_idx+1)*layers_per_group])[0]
        return hidden
    else:
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

def skeleton_model(start_layer_id,start_hidden,model,interventions,args):
    if args.model.startswith('bert'):
        core_model = model.bert
        lm_head = model.cls
    elif args.model.startswith('roberta'):
        core_model = model.roberta
        lm_head = model.lm_head
    elif args.model.startswith('deberta'):
        core_model = model.deberta
        lm_head = model.cls
        args.encoder = model.deberta.encoder
        args.rel_embeddings = model.deberta.encoder.get_rel_embedding().to(args.device)
        args = args_setup_deberta(args, start_hidden.shape[:-1])
    elif args.model.startswith('albert'):
        core_model = model.albert
        lm_head = model.predictions
        args.head_mask = model.albert.get_head_mask(None, model.albert.config.num_hidden_layers)
        args.encoder = model.albert.encoder
        args.dtype = model.albert.dtype
        args = args_setup_albert(args, start_hidden.shape[:-1])
    else:
        raise NotImplementedError("invalid model name")
    hidden = start_hidden.clone()
    with torch.no_grad():
        if args.model.startswith('albert'):
            for layer_id in range(start_layer_id, model.albert.encoder.config.num_hidden_layers):
                hidden = layer_intervention(layer_id,None,interventions,hidden,args)
        else:
            for layer_id,layer in zip(np.arange(start_layer_id,len(core_model.encoder.layer)),core_model.encoder.layer[start_layer_id:]):
                hidden = layer_intervention(layer_id,layer,interventions,hidden,args)
        logits = lm_head(hidden)
    return logits
