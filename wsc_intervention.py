import numpy as np
import torch
import pickle
import torch.nn.functional as F
import argparse
import json
import csv
from wsc_utils import CalcOutputs, EvaluatePredictions, LoadDataset, LoadModel
from model_skeleton import skeleton_model
import pandas as pd
import time

def ExtractAttnLayer(layer_id,model,args):
    if args.model.startswith('bert'):
        layer = model.bert.encoder.layer[layer_id].attention.self
    elif args.model.startswith('roberta'):
        layer = model.roberta.encoder.layer[layer_id].attention.self
    elif args.model.startswith('albert'):
        layer = model.albert.encoder.albert_layer_groups[0].albert_layers[0].attention
    else:
        raise NotImplementedError("invalid model name")
    return layer

def GetReps(layer_id,head_id,pos_type,rep_type,outputs,token_ids,args):
    assert pos_type in ['','option_1','option_2','context','masks','period','cls','sep','other']
    assert rep_type in ['layer','key','query','value','attention']
    if rep_type=='layer':
        vec = outputs[1][layer_id][0,token_ids[f'{pos_type}']]
        return vec
    elif rep_type in ['key','query','value']:
        attn_layer = ExtractAttnLayer(layer_id,model,args)
        if rep_type=='key':
            key = attn_layer.key(outputs[1][layer_id])
            vec = key[0,token_ids[f'{pos_type}']]
        elif rep_type=='query':
            query = attn_layer.query(outputs[1][layer_id])
            vec = query[0,token_ids[f'{pos_type}']]
        elif rep_type=='value':
            value = attn_layer.value(outputs[1][layer_id])
            vec = value[0,token_ids[f'{pos_type}']]
        return vec
    elif rep_type=='attention':
        mat = outputs[2][layer_id][0,head_id]
        return mat
    else:
        raise NotImplementedError(f'rep_type "{rep_type}" is not supported')

def ApplyInterventionsLayer(model,layer_id,head_id,pos_types,rep_types,outputs,token_ids,option_tokens_lists,args,verbose=False):
    interventions_all = []
    for context_id in range(2):
        interventions = {'masked_sent_1':{},'masked_sent_2':{}}
        for masked_sent_id in [1,2]:
            for rep_type in rep_types:
                if rep_type=='attention':
                    if args.test:
                        attn = GetReps(layer_id,head_id,'',rep_type,
                                    outputs[f'masked_sent_{masked_sent_id}_context_{context_id+1}'],
                                    token_ids[context_id][f'masked_sent_{masked_sent_id}'],args)
                    else:
                        attn = GetReps(layer_id,head_id,'',rep_type,
                                    outputs[f'masked_sent_{masked_sent_id}_context_{2-context_id}'],
                                    token_ids[1-context_id][f'masked_sent_{masked_sent_id}'],args)
                    interventions[f'masked_sent_{masked_sent_id}'][f'attention_{layer_id}_{head_id}'] = attn
                else:
                    for pos_type in pos_types:
                        pos = token_ids[context_id][f'masked_sent_{masked_sent_id}'][f'{pos_type}']
                        if args.test:
                            vec = GetReps(layer_id,head_id,pos_type,rep_type,
                                        outputs[f'masked_sent_{masked_sent_id}_context_{context_id+1}'],
                                        token_ids[context_id][f'masked_sent_{masked_sent_id}'],args)
                        else:
                            vec = GetReps(layer_id,head_id,pos_type,rep_type,
                                        outputs[f'masked_sent_{masked_sent_id}_context_{2-context_id}'],
                                        token_ids[1-context_id][f'masked_sent_{masked_sent_id}'],args)
                        if pos_type!='context' or not args.no_eq_len_condition:
                            assert len(pos)==len(vec)
                        if f'{rep_type}_{layer_id}' not in interventions[f'masked_sent_{masked_sent_id}']:
                            interventions[f'masked_sent_{masked_sent_id}'][f'{rep_type}_{layer_id}'] = []
                        interventions[f'masked_sent_{masked_sent_id}'][f'{rep_type}_{layer_id}'].extend([(pos,vec)])
        interventions_all.append(interventions)

    if verbose:
        for intervention in [interventions_all[0]['masked_sent_1'],interventions_all[0]['masked_sent_2'],interventions_all[1]['masked_sent_1'],interventions_all[1]['masked_sent_2']]:
            for key,value in intervention.items():
                print(key)
                for pair in value:
                    print(pair[0],pair[1].shape)

    int_logits_1_context_1 = skeleton_model(layer_id,outputs['masked_sent_1_context_1'][1][layer_id],model,interventions_all[0]['masked_sent_1'],args)
    int_logits_2_context_1 = skeleton_model(layer_id,outputs['masked_sent_2_context_1'][1][layer_id],model,interventions_all[0]['masked_sent_2'],args)
    int_logits_1_context_2 = skeleton_model(layer_id,outputs['masked_sent_1_context_2'][1][layer_id],model,interventions_all[1]['masked_sent_1'],args)
    int_logits_2_context_2 = skeleton_model(layer_id,outputs['masked_sent_2_context_2'][1][layer_id],model,interventions_all[1]['masked_sent_2'],args)

    if 'context' in pos_types and not args.test:
        token_ids_new_1 = token_ids[1]
        token_ids_new_2 = token_ids[0]
    else:
        token_ids_new_1 = token_ids[0]
        token_ids_new_2 = token_ids[1]

    option_tokens_list_1 = option_tokens_lists[0]
    option_tokens_list_2 = option_tokens_lists[1]

    choice_probs_sum_1,choice_probs_ave_1 = EvaluatePredictions(int_logits_1_context_1,int_logits_2_context_1,token_ids_new_1,option_tokens_list_1,args)
    choice_probs_sum_2,choice_probs_ave_2 = EvaluatePredictions(int_logits_1_context_2,int_logits_2_context_2,token_ids_new_2,option_tokens_list_2,args)

    return choice_probs_sum_1,choice_probs_ave_1,choice_probs_sum_2,choice_probs_ave_2


def ApplyInterventions(head,line,pos_types,rep_types,model,tokenizer,mask_id,args):
    assert int(line[head.index('option_1_word_id_1')]) < int(line[head.index('option_2_word_id_1')])
    assert int(line[head.index('option_1_word_id_2')]) < int(line[head.index('option_2_word_id_2')])

    outputs_1,token_ids_1,option_tokens_list_1,_ = CalcOutputs(head,line,1,model,tokenizer,mask_id,args)
    outputs_2,token_ids_2,option_tokens_list_2,_ = CalcOutputs(head,line,2,model,tokenizer,mask_id,args)

    if CheckNumTokens(outputs_1,outputs_2,token_ids_1,token_ids_2) or args.no_eq_len_condition:
        outputs = {}
        outputs['masked_sent_1_context_1'] = outputs_1[0]
        outputs['masked_sent_2_context_1'] = outputs_1[1]
        outputs['masked_sent_1_context_2'] = outputs_2[0]
        outputs['masked_sent_2_context_2'] = outputs_2[1]

        token_ids = [token_ids_1, token_ids_2]
        option_tokens_lists = [option_tokens_list_1,option_tokens_list_2]

        out_dict = {}
        choice_probs_sum_1, choice_probs_ave_1 = EvaluatePredictions(outputs_1[0][0],outputs_1[1][0],token_ids_1,option_tokens_list_1,args)
        choice_probs_sum_2, choice_probs_ave_2 = EvaluatePredictions(outputs_2[0][0],outputs_2[1][0],token_ids_2,option_tokens_list_2,args)
        out_dict['original'] = {}
        out_dict['original']['sum_1'] = choice_probs_sum_1
        out_dict['original']['sum_2'] = choice_probs_sum_2
        out_dict['original']['ave_1'] = choice_probs_ave_1
        out_dict['original']['ave_2'] = choice_probs_ave_2

        for layer_id in range(model.config.num_hidden_layers):
            if 'attention' in rep_types:
                assert not args.no_eq_len_condition
                for head_id in range(model.config.num_attention_heads):
                    int_choice_probs_sum_1,int_choice_probs_ave_1,int_choice_probs_sum_2,int_choice_probs_ave_2 = ApplyInterventionsLayer(model,layer_id,head_id,pos_types,rep_types,outputs,token_ids,option_tokens_lists,args)
                    out_dict[f'layer_{layer_id}_{head_id}'] = {}
                    out_dict[f'layer_{layer_id}_{head_id}']['sum_1'] = int_choice_probs_sum_1
                    out_dict[f'layer_{layer_id}_{head_id}']['sum_2'] = int_choice_probs_sum_2
                    out_dict[f'layer_{layer_id}_{head_id}']['ave_1'] = int_choice_probs_ave_1
                    out_dict[f'layer_{layer_id}_{head_id}']['ave_2'] = int_choice_probs_ave_2
            else:
                int_choice_probs_sum_1,int_choice_probs_ave_1,int_choice_probs_sum_2,int_choice_probs_ave_2 = ApplyInterventionsLayer(model,layer_id,0,pos_types,rep_types,outputs,token_ids,option_tokens_lists,args)
                out_dict[f'layer_{layer_id}'] = {}
                out_dict[f'layer_{layer_id}']['sum_1'] = int_choice_probs_sum_1
                out_dict[f'layer_{layer_id}']['sum_2'] = int_choice_probs_sum_2
                out_dict[f'layer_{layer_id}']['ave_1'] = int_choice_probs_ave_1
                out_dict[f'layer_{layer_id}']['ave_2'] = int_choice_probs_ave_2
        return out_dict
    else:
        return 'number of tokens did not match'

def CheckNumTokens(outputs_1,outputs_2,token_ids_1,token_ids_2):
    if outputs_1[0][0].shape[1]==outputs_2[0][0].shape[1] and outputs_1[1][0].shape[1]==outputs_2[1][0].shape[1]:
        for sent_id in [1,2]:
            for feature in ['option_1','option_2','context','masks','other','period','cls','sep']:
                assert torch.all(token_ids_1[f'masked_sent_{sent_id}'][feature]==token_ids_2[f'masked_sent_{sent_id}'][feature])
        return True
    else:
        return False


if __name__=='__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--dataset', type = str, required = True, choices=['superglue','winogrande'])
    parser.add_argument('--stimuli', type = str,
                        choices=['original','control_gender','control_number','control_combined'],
                        default='original')
    parser.add_argument('--size', type = str, choices=['xs','s','m','l','xl','debiased'])
    parser.add_argument('--core_id', type = int, default=0)
    parser.add_argument('--rep_type', type = str, required = True)
    parser.add_argument('--pos_type', type = str)
    parser.add_argument('--test',dest='test',action='store_true')
    parser.add_argument('--no_eq_len_condition',dest='no_eq_len_condition',action='store_true')
    parser.set_defaults(test=False,no_eq_len_condition=False)
    args = parser.parse_args()
    # context and masks perturbations should be the last since they may change the sentence length
    #assert args.rep_type in ['layer','query','key','value','layer-query','key-value','layer-query-key-value']
    #assert args.pos_type in ['option_1','option_2','option_1-option_2','context','masks','period','cls','sep','cls-sep','cls-period-sep','option_1-option_2-context','other']
    print(f'running with {args}')

    if args.pos_type is None:
        assert args.rep_type=='attention'

    if args.test:
        test_id = '_test'
    else:
        test_id = ''

    head,text = LoadDataset(args)
    model, tokenizer, mask_id, args = LoadModel(args)

    out_dict = {}
    for line in text[:500]:
        if args.pos_type is None:
            results = ApplyInterventions(head,line,[],args.rep_type.split('-'),model,tokenizer,mask_id,args)
        else:
            results = ApplyInterventions(head,line,args.pos_type.split('-'),args.rep_type.split('-'),model,tokenizer,mask_id,args)
        if type(results) is str:
            continue
        else:
            out_dict[line[head.index('pair_id')]] = results

    if args.pos_type is None:
        if args.dataset=='superglue':
            with open(f'datafile/superglue_wsc_intervention_{args.rep_type}_{args.model}_{args.stimuli}{test_id}.pkl','wb') as f:
                pickle.dump(out_dict,f)
        elif args.dataset=='winogrande':
            with open(f'datafile/winogrande_{args.size}_intervention_{args.rep_type}_{args.model}{test_id}.pkl','wb') as f:
                pickle.dump(out_dict,f)
    else:
        if args.dataset=='superglue':
            with open(f'datafile/superglue_wsc_intervention_{args.pos_type}_{args.rep_type}_{args.model}_{args.stimuli}{test_id}.pkl','wb') as f:
                pickle.dump(out_dict,f)
        elif args.dataset=='winogrande':
            with open(f'datafile/winogrande_{args.size}_intervention_{args.pos_type}_{args.rep_type}_{args.model}{test_id}.pkl','wb') as f:
                pickle.dump(out_dict,f)

    print(f'Time it took: {time.time()-start}')
    print(f'# sentences processed: {len(list(out_dict.keys()))}\n')
