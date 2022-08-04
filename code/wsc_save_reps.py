# export DATA_PATH='YOUR PATH TO DATA'
import numpy as np
import torch
import pickle
import torch.nn.functional as F
import argparse
import json
import csv
from wsc_utils import CalcOutputs, LoadDataset, LoadModel
import pandas as pd
import os

def token_ids_to_cpu(token_ids):
    new_token_ids = {}
    new_token_ids['pron_id'] = token_ids['pron_id'].to('cpu')
    for masked_sent_id in [1,2]:
        new_token_ids[f'masked_sent_{masked_sent_id}'] = {}
        for pos_id,pos in token_ids[f'masked_sent_{masked_sent_id}'].items():
            new_token_ids[f'masked_sent_{masked_sent_id}'][pos_id] = pos.to('cpu')
    return new_token_ids

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--dataset', type = str, required = True, choices=['superglue','winogrande'])
    parser.add_argument('--stimuli', type = str,
                        choices=['original','control_gender','control_number','control_combined'],
                        default='original')
    parser.add_argument('--size', type = str, choices=['xs','s','m','l','xl','debiased'])
    parser.add_argument('--core_id', type = int, default=0)
    args = parser.parse_args()
    print(f'running with {args}')

    head,text = LoadDataset(args)
    model, tokenizer, mask_id, args = LoadModel(args)

    out_dict = {}
    for line in text[:100]:
        out_dict[line[head.index('pair_id')]] = {}
        outputs_1, token_ids_1, option_tokens_list_1, masked_sents_1 = CalcOutputs(head,line,1,model,tokenizer,mask_id,args)
        #out_dict[line[head.index('pair_id')]]['probs_1'] = [output[0].to('cpu') for output in outputs_1]
        out_dict[line[head.index('pair_id')]]['reps_1'] = [[layer.to('cpu') for layer in output[1]] for output in outputs_1]
        #out_dict[line[head.index('pair_id')]]['attn_1'] = [[layer.to('cpu') for layer in output[2]] for output in outputs_1]
        out_dict[line[head.index('pair_id')]]['token_ids_1'] = token_ids_to_cpu(token_ids_1)
        #out_dict[line[head.index('pair_id')]]['option_tokens_list_1'] = [tokens.to('cpu') for tokens in option_tokens_list_1]
        #out_dict[line[head.index('pair_id')]]['masked_sents_1'] = [sent.to('cpu') for sent in masked_sents_1]

        outputs_2, token_ids_2, option_tokens_list_2, masked_sents_2 = CalcOutputs(head,line,2,model,tokenizer,mask_id,args)
        #out_dict[line[head.index('pair_id')]]['probs_2'] = [output[0].to('cpu') for output in outputs_2]
        out_dict[line[head.index('pair_id')]]['reps_2'] = [[layer.to('cpu') for layer in output[1]] for output in outputs_2]
        #out_dict[line[head.index('pair_id')]]['attn_2'] = [[layer.to('cpu') for layer in output[2]] for output in outputs_2]
        out_dict[line[head.index('pair_id')]]['token_ids_2'] = token_ids_to_cpu(token_ids_2)
        #out_dict[line[head.index('pair_id')]]['option_tokens_list_2'] = [tokens.to('cpu') for tokens in option_tokens_list_2]
        #out_dict[line[head.index('pair_id')]]['masked_sents_2'] = [sent.to('cpu') for sent in masked_sents_2]

        outputs_3, token_ids_3, option_tokens_list_3, masked_sents_3 = CalcOutputs(head,line,1,model,tokenizer,mask_id,args,mask_context=True)
        #out_dict[line[head.index('pair_id')]]['probs_3'] = [output[0].to('cpu') for output in outputs_3]
        out_dict[line[head.index('pair_id')]]['reps_3'] = [[layer.to('cpu') for layer in output[1]] for output in outputs_3]
        #out_dict[line[head.index('pair_id')]]['attn_3'] = [[layer.to('cpu') for layer in output[2]] for output in outputs_3]
        out_dict[line[head.index('pair_id')]]['token_ids_3'] = token_ids_to_cpu(token_ids_3)
        #out_dict[line[head.index('pair_id')]]['option_tokens_list_3'] = [tokens.to('cpu') for tokens in option_tokens_list_3]
        #out_dict[line[head.index('pair_id')]]['masked_sents_3'] = [sent.to('cpu') for sent in masked_sents_3]

    if args.dataset=='superglue':
        with open(f'{os.environ.get("DATA_PATH")}/superglue_wsc_output_all_{args.model}_{args.stimuli}.pkl','wb') as f:
            pickle.dump(out_dict,f)
    elif args.dataset=='winogrande':
        with open(f'{os.environ.get("DATA_PATH")}/winogrande_{args.size}_output_all_{args.model}.pkl','wb') as f:
            pickle.dump(out_dict,f)
