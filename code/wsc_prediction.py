# export MY_DATA_PATH='YOUR PATH TO DATA FILES'
import numpy as np
import torch
import pickle
import torch.nn.functional as F
import argparse
import json
import csv
from wsc_utils import calc_outputs, evaluate_predictions, load_dataset, load_model
import pandas as pd
import os
import time

def calc_prediction(head,line,sent_id,model,tokenizer,mask_id,args,mask_context=False):
    outputs, token_ids, option_tokens_list, masked_sents = calc_outputs(head,line,sent_id,model,tokenizer,mask_id,args,mask_context=mask_context)
    if 'bert' in args.model:
        tokens_list = option_tokens_list
    elif 'gpt2' in args.model:
        tokens_list = [masked_sent[0][1:] for masked_sent in masked_sents]
    return evaluate_predictions(outputs[0][0],outputs[1][0],token_ids['pron_id'],tokens_list,args)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--dataset', type = str, required = True, choices=['superglue','winogrande','combined'])
    parser.add_argument('--stimuli', type = str,
                        choices=['original','control','synonym_1','synonym_2'],
                        #'original_verb','control_combined','control_combined_verb','synonym_verb'],
                        default='original')
    parser.add_argument('--size', type = str, choices=['xs','s','m','l','xl','debiased'])
    parser.add_argument('--core_id', type = int, default=0)
    parser.add_argument('--mask_context',dest='mask_context',action='store_true')
    parser.add_argument('--no_mask',dest='no_mask',action='store_true')
    parser.set_defaults(mask_context=False,no_mask=False)
    args = parser.parse_args()
    print(f'running with {args}')

    head,text = load_dataset(args)
    model, tokenizer, mask_id, args = load_model(args)
    mask_context_id = '_mask_context' if args.mask_context else ''

    os.makedirs(f'{os.environ.get("MY_DATA_PATH")}/prediction/',exist_ok=True)

    dataset_name = args.dataset + f'_{args.size}' if args.dataset == 'winogrande' else args.dataset
    out_file_name = f'{os.environ.get("MY_DATA_PATH")}/prediction/{dataset_name}_{args.stimuli}{mask_context_id}_prediction_{args.model}'

    start = time.time()
    with open(f'{out_file_name}.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(head+['sum_1','sum_2','ave_1','ave_2'])
        for line in text:
            choice_probs_sum_1, choice_probs_ave_1 = calc_prediction(head,line,1,model,tokenizer,mask_id,args,mask_context=args.mask_context)
            choice_probs_sum_2, choice_probs_ave_2 = calc_prediction(head,line,2,model,tokenizer,mask_id,args,mask_context=args.mask_context)
            writer.writerow(line+[
                                choice_probs_sum_1[0]-choice_probs_sum_1[1],
                                choice_probs_sum_2[0]-choice_probs_sum_2[1],
                                choice_probs_ave_1[0]-choice_probs_ave_1[1],
                                choice_probs_ave_2[0]-choice_probs_ave_2[1]])

    print(f'{len(text)} sentences done')
    print(f'Time: {time.time()-start}')
