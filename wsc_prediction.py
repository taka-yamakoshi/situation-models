import numpy as np
import torch
import pickle
import torch.nn.functional as F
import argparse
import json
import csv
from wsc_utils import CalcOutputs, EvaluatePredictions, LoadDataset, LoadModel
import pandas as pd

def CalcPrediction(head,line,sent_id,model,tokenizer,mask_id,args,mask_context=False):
    outputs, token_ids, option_tokens_list, masked_sents = CalcOutputs(head,line,sent_id,model,tokenizer,mask_id,args,mask_context=mask_context)
    if 'bert' in args.model:
        tokens_list = option_tokens_list
    elif 'gpt2' in args.model:
        tokens_list = [masked_sent[0] for masked_sent in masked_sents]
    return EvaluatePredictions(outputs[0][0],outputs[1][0],token_ids,tokens_list,args)

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
    for line in text[:500]:
        choice_probs_sum_1, choice_probs_ave_1 = CalcPrediction(head,line,1,model,tokenizer,mask_id,args)
        choice_probs_sum_2, choice_probs_ave_2 = CalcPrediction(head,line,2,model,tokenizer,mask_id,args)
        choice_probs_sum_3, choice_probs_ave_3 = CalcPrediction(head,line,1,model,tokenizer,mask_id,args,mask_context=True)

        out_dict[line[head.index('pair_id')]] = {}
        out_dict[line[head.index('pair_id')]]['sum_1'] = choice_probs_sum_1
        out_dict[line[head.index('pair_id')]]['sum_2'] = choice_probs_sum_2
        out_dict[line[head.index('pair_id')]]['sum_3'] = choice_probs_sum_3

        out_dict[line[head.index('pair_id')]]['ave_1'] = choice_probs_ave_1
        out_dict[line[head.index('pair_id')]]['ave_2'] = choice_probs_ave_2
        out_dict[line[head.index('pair_id')]]['ave_3'] = choice_probs_ave_3

    '''
    if args.dataset=='superglue':
        with open(f'datafile/superglue_wsc_prediction_{args.model}_{args.stimuli}.pkl','wb') as f:
            pickle.dump(out_dict,f)
    elif args.dataset=='winogrande':
        with open(f'datafile/winogrande_{args.size}_prediction_{args.model}.pkl','wb') as f:
            pickle.dump(out_dict,f)
    '''

    data_list = []
    for line in text[:500]:
        pair_id = line[head.index('pair_id')]
        pred_data = out_dict[pair_id]
        data_list.append(line+[pred_data['sum_1'][0]-pred_data['sum_1'][1],
                               pred_data['sum_2'][0]-pred_data['sum_2'][1],
                               pred_data['sum_3'][0]-pred_data['sum_3'][1],
                               pred_data['ave_1'][0]-pred_data['ave_1'][1],
                               pred_data['ave_2'][0]-pred_data['ave_2'][1],
                               pred_data['ave_3'][0]-pred_data['ave_3'][1]])
    df = pd.DataFrame(data_list,columns=[head+['sum_1','sum_2','sum_3','ave_1','ave_2','ave_3']])
    if args.dataset=='superglue':
        df.to_csv(f'datafile/superglue_wsc_prediction_{args.model}_{args.stimuli}.csv')
    elif args.dataset=='winogrande':
        df.to_csv(f'datafile/winogrande_{args.size}_prediction_{args.model}.csv')
