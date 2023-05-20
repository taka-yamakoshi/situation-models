# export MY_DATA_PATH='YOUR PATH TO DATA FILES'
import numpy as np
import torch
import pickle
import torch.nn.functional as F
import argparse
import json
import csv
from wsc_utils import calc_outputs, evaluate_predictions, load_dataset, load_model, get_reps, extract_QKV, evaluate_QKV
from model_skeleton import skeleton_model, extract_attn_layer
from wsc_attention import evaluate_attention,convert_to_numpy
from wsc_intervention import apply_interventions, evaluate_results
import pandas as pd
import time
import math
import os

def calc_seq_len(head,line,tokenizer):
    sent_1,sent_2 = line[head.index('sent_1')],line[head.index('sent_2')]
    tokenized_sent_1 = tokenizer(sent_1,return_tensors='pt')['input_ids']
    tokenized_sent_2 = tokenizer(sent_1,return_tensors='pt')['input_ids']
    assert tokenized_sent_1.shape[1]==tokenized_sent_2.shape[1]
    return tokenized_sent_1.shape[1]

if __name__=='__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--dataset', type = str, required = True, choices=['superglue','winogrande','combined'])
    parser.add_argument('--pair_id', type = str, required = True)
    parser.add_argument('--stimuli', type = str,
                        choices=['original','control',
                                'original_verb','control_combined_verb','synonym_verb'],
                        default='original')
    parser.add_argument('--size', type = str, choices=['xs','s','m','l','xl','debiased'])
    parser.add_argument('--core_id', type = int, default=0)
    parser.add_argument('--layer', type = str, default='all')
    parser.add_argument('--head', type = str, default='all')
    parser.add_argument('--intervention_type',type=str,default='swap',
                        choices=['swap'])
    parser.add_argument('--test',dest='test',action='store_true')
    parser.add_argument('--no_eq_len_condition',dest='no_eq_len_condition',action='store_true')
    parser.add_argument('--no_mask',dest='no_mask',action='store_true')
    parser.add_argument('--mask_context',dest='mask_context',action='store_true')
    parser.set_defaults(test=False,no_eq_len_condition=False,cascade=False,multihead=False,no_mask=False,mask_context=False)
    args = parser.parse_args()
    print(f'running with {args}')

    if args.test:
        test_id = '_test'
    else:
        test_id = ''
    if args.mask_context:
        mask_context_id = '_mask_context'
    else:
        mask_context_id = ''

    head,text = load_dataset(args)
    pair_ids = [row[head.index('pair_id')] for row in text]
    assert args.pair_id in pair_ids and np.sum([args.pair_id==element for element in pair_ids])==1

    model, tokenizer, mask_id, args = load_model(args)
    args.num_layers = model.config.num_hidden_layers
    attn_layer = extract_attn_layer(0,model,args)
    args.num_heads = attn_layer.num_attention_heads
    args.head_dim = attn_layer.attention_head_size

    os.makedirs(f'{os.environ.get("MY_DATA_PATH")}/intervention_indiv/',exist_ok=True)
    dataset_name = args.dataset + f'_{args.size}' if args.dataset == 'winogrande' else args.dataset

    out_file_name = f'{os.environ.get("MY_DATA_PATH")}/intervention_indiv/'\
                    +f'{dataset_name}_{args.stimuli}{mask_context_id}_intervention_{args.intervention_type}'\
                    +f'_indiv_{args.pair_id}_{args.model}'\
                    +f'_layer_{args.layer}_head_{args.head}{test_id}'

    with open(f'{out_file_name}.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(head+['interv_type','rep','pos','cascade','multihead',
                                'layer_id','head_id','original_score','score','effect_ave',
                                'original_1','original_2','interv_1','interv_2','effect_1','effect_2',
                                *[f'logprob_ave_option_{option_id+1}_sent_1' for option_id in range(2)],
                                *[f'logprob_ave_option_{option_id+1}_sent_2' for option_id in range(2)],
                                *[f'logprob_sum_option_{option_id+1}_sent_1' for option_id in range(2)],
                                *[f'logprob_sum_option_{option_id+1}_sent_2' for option_id in range(2)],
                                *[f'masks-option-diff_{head_id}' for head_id in range(args.num_heads)],
                                *[f'masks-option-diff_effect_{head_id}' for head_id in range(args.num_heads)],
                                *[f'masks-qry-dist_effect_{head_id}' for head_id in range(args.num_heads)],
                                *[f'masks-qry-cos_effect_{head_id}' for head_id in range(args.num_heads)],
                                *[f'options-key-dist_effect_{head_id}' for head_id in range(args.num_heads)],
                                *[f'options-key-cos_effect_{head_id}' for head_id in range(args.num_heads)]])

        line = text[pair_ids.index(args.pair_id)]
        seq_len = calc_seq_len(head,line,tokenizer)

        rep_types = ['layer-query-key-value','z_rep','value']
        cascade_types = [False,False,False]
        multihead_types = [True,False,False]
        for rep,cascade_id,multihead_id in zip(rep_types, cascade_types, multihead_types):
            args.cascade, args.multihead = cascade_id, multihead_id
            for pos in range(seq_len):
                results = apply_interventions(head,line,[f'token_{pos}'],rep.split('-'),model,tokenizer,mask_id,args)
                if type(results) is str:
                    raise NotImplementedError("Sequence lengths do not match")
                else:
                    original_results = evaluate_results(results['original'],0,args)

                    original_qry_dist,original_qry_cos = evaluate_QKV('qry',results['original'],results['original'],0,0,args)
                    original_key_dist,original_key_cos = evaluate_QKV('key',results['original'],results['original'],0,0,args)
                    for layer_id in range(args.num_layers):
                        result_dict = results[f'layer_{layer_id}']
                        for head_id in range(args.num_heads):
                            interv_results = evaluate_results(result_dict,head_id,args)

                            effect_1 = original_results['llr_1']-interv_results['llr_1']
                            effect_2 = interv_results['llr_2']-original_results['llr_2']
                            effect_attn_1 = original_results['attn_1']-interv_results['attn_1']
                            effect_attn_2 = interv_results['attn_2']-original_results['attn_2']

                            interv_qry_dist,interv_qry_cos = evaluate_QKV('qry',result_dict,results['original'],head_id,0,args)
                            interv_key_dist,interv_key_cos = evaluate_QKV('key',result_dict,results['original'],head_id,0,args)
                            assert len(interv_qry_dist)==args.num_heads and len(interv_qry_cos)==args.num_heads
                            assert len(interv_key_dist)==args.num_heads and len(interv_key_cos)==args.num_heads

                            writer.writerow(line+['interv',rep,pos,cascade_id,multihead_id,
                                                    layer_id,head_id,original_results['score'],interv_results['score'],(effect_1+effect_2)/2,
                                                    original_results['llr_1'],original_results['llr_2'],
                                                    interv_results['llr_1'],interv_results['llr_2'],effect_1,effect_2,
                                                    *[interv_results[f'logprob_ave_option_{option_id+1}_sent_1'] for option_id in range(2)],
                                                    *[interv_results[f'logprob_ave_option_{option_id+1}_sent_2'] for option_id in range(2)],
                                                    *[interv_results[f'logprob_sum_option_{option_id+1}_sent_1'] for option_id in range(2)],
                                                    *[interv_results[f'logprob_sum_option_{option_id+1}_sent_2'] for option_id in range(2)],
                                                    *list((interv_results['attn_1']-interv_results['attn_2'])/2),
                                                    *list((effect_attn_1+effect_attn_2)/2),
                                                    *list(original_qry_dist-interv_qry_dist),
                                                    *list(interv_qry_cos-original_qry_cos),
                                                    *list(original_key_dist-interv_key_dist),
                                                    *list(interv_key_cos-original_key_cos)])
                            writer.writerow(line+['original',rep,pos,cascade_id,multihead_id,
                                                    layer_id,head_id,original_results['score'],original_results['score'],0.0,
                                                    original_results['llr_1'],original_results['llr_2'],
                                                    original_results['llr_1'],original_results['llr_2'],0.0,0.0,
                                                    *[original_results[f'logprob_ave_option_{option_id+1}_sent_1'] for option_id in range(2)],
                                                    *[original_results[f'logprob_ave_option_{option_id+1}_sent_2'] for option_id in range(2)],
                                                    *[original_results[f'logprob_sum_option_{option_id+1}_sent_1'] for option_id in range(2)],
                                                    *[original_results[f'logprob_sum_option_{option_id+1}_sent_2'] for option_id in range(2)],
                                                    *list((original_results['attn_1']-original_results['attn_2'])/2),
                                                    *[0.0 for _ in range(args.num_heads)],
                                                    *[0.0 for _ in range(args.num_heads)],
                                                    *[0.0 for _ in range(args.num_heads)],
                                                    *[0.0 for _ in range(args.num_heads)],
                                                    *[0.0 for _ in range(args.num_heads)]])

    print(f'Time it took: {time.time()-start}')
