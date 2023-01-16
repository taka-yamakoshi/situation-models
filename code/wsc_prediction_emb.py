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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--dataset', type = str, required = True, choices=['superglue','winogrande','combined'])
    parser.add_argument('--stimuli', type = str,
                        choices=['original','control','synonym_1','synonym_2','original_verb'],
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
    out_file_name = f'{os.environ.get("MY_DATA_PATH")}/prediction/{dataset_name}_{args.stimuli}{mask_context_id}_prediction_emb_{args.model}'

    if args.stimuli=='original_verb' or args.dataset=='combined':
        has_verb = True
    else:
        has_verb = False

    start = time.time()
    with open(f'{out_file_name}.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(head+['sent_id','layer_id','metric','baseline','pred'])
        for line in text:
            for sent_id in [1,2]:
                output, output_token_ids, option_tokens_list, masked_sent = calc_outputs(head,line,sent_id,model,tokenizer,mask_id,
                                                                                         args,mask_context=False,output_for_attn=False,use_skeleton=False)
                for layer_id in range(len(output[0][1])):
                    corr_context_list = []
                    dist_context_list = []
                    if has_verb:
                        corr_verb_list = []
                        dist_verb_list = []
                    for masked_sent_id in range(2):
                        context_vec = output[masked_sent_id][1][layer_id][0][output_token_ids[f'masked_sent_{masked_sent_id+1}']['context']].mean(dim=0)
                        if has_verb:
                            verb_vec = output[masked_sent_id][1][layer_id][0][output_token_ids[f'masked_sent_{masked_sent_id+1}']['verb']].mean(dim=0)
                        option_vec = output[masked_sent_id][1][layer_id][0][output_token_ids[f'masked_sent_{masked_sent_id+1}'][f'option_{masked_sent_id+1}']].mean(dim=0)

                        corr_context = torch.corrcoef(torch.stack((context_vec,option_vec)))[0,1]
                        dist_context = torch.linalg.norm(context_vec-option_vec)
                        corr_context_list.append(corr_context)
                        dist_context_list.append(dist_context)

                        if has_verb:
                            corr_verb = torch.corrcoef(torch.stack((verb_vec,option_vec)))[0,1]
                            dist_verb = torch.linalg.norm(verb_vec-option_vec)
                            corr_verb_list.append(corr_verb)
                            dist_verb_list.append(dist_verb)

                    pred_corr_context = corr_context_list[0] - corr_context_list[1]
                    pred_dist_context = dist_context_list[1] - dist_context_list[0]
                    if has_verb:
                        pred_corr_verb = corr_verb_list[0] - corr_verb_list[1]
                        pred_dist_verb = dist_verb_list[1] - dist_verb_list[0]

                    writer.writerow(line+[sent_id,layer_id,'corr','context',pred_corr_context.item()])
                    writer.writerow(line+[sent_id,layer_id,'dist','context',pred_dist_context.item()])
                    if has_verb:
                        writer.writerow(line+[sent_id,layer_id,'corr','verb',pred_corr_verb.item()])
                        writer.writerow(line+[sent_id,layer_id,'dist','verb',pred_dist_verb.item()])

    print(f'{len(text)} sentences done')
    print(f'Time: {time.time()-start}')
