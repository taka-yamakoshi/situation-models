import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import argparse
import json
import csv
from wsc_utils import CalcOutputs, LoadDataset, LoadModel

def convert_to_numpy(attn):
    return np.array([layer.to('cpu').squeeze().detach().numpy() for layer in attn])

def calc_attention_rollout(attn):
    full_attn = 0.5*attn+0.5*np.eye(attn.shape[-1])[None,None,...]
    ave_attn = full_attn.mean(axis=1)
    attn_rollout = [full_attn[0]]
    ave_attn_rollout = [ave_attn[0]]
    for layer,ave_layer in zip(full_attn[1:],ave_attn[1:]):
        layer_rollout = np.array([list(head@ave_attn_rollout[-1]) for head in layer])
        attn_rollout.append(layer_rollout)
        ave_attn_rollout.append(ave_layer@ave_attn_rollout[-1])
    return np.array(attn_rollout)

def calc_attn_norm(model,hidden,attention,args):
    assert len(hidden)-1==len(attention)
    assert attention[0].shape[-2]==attention[0].shape[-1]
    num_layers = len(attention)
    num_heads = len(attention[0].squeeze())
    seq_len = attention[0].shape[-1]
    head_dim = hidden[0].shape[-1]//num_heads
    assert num_heads*head_dim==hidden[0].shape[-1]
    attn_norm = np.empty((num_layers,num_heads,seq_len,seq_len))
    for layer_id in range(num_layers):
        if args.model.startswith('bert'):
            value = model.bert.encoder.layer[layer_id].attention.self.value(hidden[layer_id])
        elif args.model.startswith('roberta'):
            value = model.roberta.encoder.layer[layer_id].attention.self.value(hidden[layer_id])
        elif args.model.startswith('albert'):
            value = model.albert.encoder.albert_layer_groups[0].albert_layers[0].attention.value(hidden[layer_id])
        assert value.shape==hidden[layer_id].shape
        for head_id in range(num_heads):
            head_value = value[0,:,head_dim*head_id:head_dim*(head_id+1)]
            head_attn = attention[layer_id][0][head_id]
            alpha_value = head_attn[...,None]*head_value[None,...]
            attn_norm[layer_id,head_id] = torch.linalg.norm(alpha_value,dim=-1).to('cpu').detach().numpy()
    return attn_norm

def EvaluateAttention(attention,token_ids):
    pron_token_id = token_ids['pron_id'].to('cpu')
    attn_choice_1 = attention[:,:,pron_token_id,token_ids['option_1'].to('cpu')]
    attn_choice_2 = attention[:,:,pron_token_id,token_ids['option_2'].to('cpu')]
    attn_context = attention[:,:,pron_token_id,token_ids['context'].to('cpu')]
    attn_period = attention[:,:,pron_token_id,token_ids['period'].to('cpu')]

    if len(attn_choice_1.shape)==3:
        attn_choice_1 = attn_choice_1.sum(axis=-1)
    if len(attn_choice_2.shape)==3:
        attn_choice_2 = attn_choice_2.sum(axis=-1)
    if len(attn_context.shape)==3:
        attn_context = attn_context.sum(axis=-1)
    if len(attn_period.shape)==3:
        attn_period = attn_period.sum(axis=-1)

    return attn_choice_1,attn_choice_2,attn_context,attn_period

def CalcAttn(head,line,sent_id,model,tokenizer,mask_id,args,mask_context=False):
    output, token_ids, option_tokens_list, masked_sent = CalcOutputs(head,line,sent_id,model,tokenizer,mask_id,args,mask_context=mask_context, output_for_attn=True)
    if args.norm:
        attention = calc_attn_norm(model,output[1],output[2],args)
    else:
        attention = convert_to_numpy(output[2])
        if args.roll_out:
            attention = calc_attention_rollout(attention)

    return EvaluateAttention(attention,token_ids)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--dataset', type = str, required = True, choices=['superglue','winogrande'])
    parser.add_argument('--stimuli', type = str,
                        choices=['original','control_gender','control_number','control_combined'],
                        default='original')
    parser.add_argument('--size', type = str, choices=['xs','s','m','l','xl','debiased'])
    parser.add_argument('--core_id', type = int, default=0)
    parser.add_argument('--roll_out',dest='roll_out',action='store_true')
    parser.add_argument('--norm',dest='norm',action='store_true')
    parser.add_argument('--no_mask',dest='no_mask',action='store_true')
    parser.set_defaults(roll_out=False)
    parser.set_defaults(norm=False)
    parser.set_defaults(no_mask=False)
    args = parser.parse_args()
    print(f'running with {args}')

    if args.roll_out:
        roll_out_id = '_roll_out'
    else:
        roll_out_id = ''

    if args.norm:
        norm_id = '_norm'
    else:
        norm_id = ''

    if args.no_mask:
        no_mask_id = '_no_mask'
    else:
        no_mask_id = ''

    head,text = LoadDataset(args)
    model, tokenizer, mask_id, args = LoadModel(args)

    out_dict = {}
    for line in text[:500]:
        attn_choice_1_context_1,attn_choice_2_context_1,attn_context_context_1,attn_period_context_1 = CalcAttn(head,line,1,model,tokenizer,mask_id,args)
        attn_choice_1_context_2,attn_choice_2_context_2,attn_context_context_2,attn_period_context_2 = CalcAttn(head,line,2,model,tokenizer,mask_id,args)

        out_dict[line[head.index('pair_id')]] = {}
        out_dict[line[head.index('pair_id')]]['choice_1_context_1'] = attn_choice_1_context_1
        out_dict[line[head.index('pair_id')]]['choice_2_context_1'] = attn_choice_2_context_1
        out_dict[line[head.index('pair_id')]]['choice_1_context_2'] = attn_choice_1_context_2
        out_dict[line[head.index('pair_id')]]['choice_2_context_2'] = attn_choice_2_context_2
        out_dict[line[head.index('pair_id')]]['context_1'] = attn_context_context_1
        out_dict[line[head.index('pair_id')]]['context_2'] = attn_context_context_2
        out_dict[line[head.index('pair_id')]]['period_1'] = attn_period_context_1
        out_dict[line[head.index('pair_id')]]['period_2'] = attn_period_context_2

    if args.dataset=='superglue':
        with open(f'datafile/superglue_wsc_attention{norm_id}_{args.model}_{args.stimuli}{roll_out_id}{no_mask_id}.pkl','wb') as f:
            pickle.dump(out_dict,f)
    elif args.dataset=='winogrande':
        with open(f'datafile/winogrande_{args.size}_attention{norm_id}_{args.model}{roll_out_id}{no_mask_id}.pkl','wb') as f:
            pickle.dump(out_dict,f)
