# export MY_DATA_PATH='YOUR PATH TO DATA FILES'
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import argparse
import json
import csv
import os
from wsc_utils import calc_outputs, load_dataset, load_model

def convert_to_numpy(attn):
    return np.array([layer.to('cpu').detach().numpy() for layer in attn])

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

def evaluate_attention(attention,token_ids,prediction_task=False,target_layer_id=None):
    attn_dict = {}
    if prediction_task:
        out_pos = 'masks'
    else:
        out_pos = 'pron_id'
    for in_pos in ['option_1','option_2','options','context','period','cls','sep']:
        attn_dict[f'{out_pos}-{in_pos}'] = extract_attention(attention,in_pos,out_pos,token_ids,target_layer_id)
    for out_pos in ['option_1','option_2','options']:
        for in_pos in ['masks','context','period','cls','sep']:
            attn_dict[f'{out_pos}-{in_pos}'] = extract_attention(attention,in_pos,out_pos,token_ids,target_layer_id)
    return attn_dict

def extract_attention(attn,in_pos,out_pos,token_ids,target_layer_id):
    #attn.shape=(num_layers,batch_size,num_heads,seq_len,seq_len)
    assert len(attn.shape)==5 and attn.shape[3]==attn.shape[4]
    attn = attn[:,:,:,:,token_ids[in_pos].to('cpu')]
    if len(token_ids[in_pos])>1:
        attn = attn.sum(axis=-1)
    assert len(attn.shape)==4
    attn = attn[:,:,:,token_ids[out_pos].to('cpu')]
    if len(token_ids[out_pos])>1:
        attn = attn.mean(axis=-1)
    assert len(attn.shape)==3
    if target_layer_id is not None:
        return attn[target_layer_id,:,:].squeeze()
    else:
        return attn.squeeze()

def calc_attn(head,line,sent_id,model,tokenizer,mask_id,args,mask_context=False):
    output, token_ids, option_tokens_list, masked_sent = calc_outputs(head,line,sent_id,model,tokenizer,mask_id,args,mask_context=mask_context, output_for_attn=True)
    if args.norm:
        attention = calc_attn_norm(model,output[1],output[2],args)
    else:
        attention = convert_to_numpy(output[2])
        if args.roll_out:
            attention = calc_attention_rollout(attention)

    return evaluate_attention(attention,token_ids)

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

    roll_out_id = '_roll_out' if args.roll_out else ''
    norm_id = '_norm' if args.norm else ''
    no_mask_id = '_no_mask' if args.no_mask else ''

    head,text = load_dataset(args)
    model, tokenizer, mask_id, args = load_model(args)

    out_dict = {}
    for line in text:
        attn_dict_1 = calc_attn(head,line,1,model,tokenizer,mask_id,args)
        attn_dict_2 = calc_attn(head,line,2,model,tokenizer,mask_id,args)

        out_dict[line[head.index('pair_id')]] = {}
        for sent_id,attn_dict in zip([1,2],[attn_dict_1,attn_dict_2]):
            for key,val in attn_dict.items():
                out_dict[line[head.index('pair_id')]][f'{key}_sent_{sent_id}'] = val

    os.makedirs(f'{os.environ.get("MY_DATA_PATH")}/attention',exist_ok=True)
    dataset_name = args.dataset + f'_{args.size}' if args.dataset == 'winogrande' else args.dataset

    out_file_name = f'{os.environ.get("MY_DATA_PATH")}/attention/'\
                    +f'{dataset_name}_{args.stimuli}_attention{norm_id}_{args.model}{roll_out_id}{no_mask_id}.pkl'
    with open(f'{out_file_name}','wb') as f:
        pickle.dump(out_dict,f)
