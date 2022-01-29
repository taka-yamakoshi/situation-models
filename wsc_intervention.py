# export DATA_PATH='YOUR PATH TO DATA FILES'
import numpy as np
import torch
import pickle
import torch.nn.functional as F
import argparse
import json
import csv
from wsc_utils import CalcOutputs, EvaluatePredictions, LoadDataset, LoadModel, GetReps
from model_skeleton import skeleton_model, ExtractAttnLayer
from wsc_attention import EvaluateAttention,convert_to_numpy
import pandas as pd
import time
import math
import os

def CreateInterventions(model,interventions,layer_id,head_id,pos_types,rep_types,outputs,token_ids,args,verbose=False):
    if len(interventions)==0:
        interventions = [{'masked_sent_1':{},'masked_sent_2':{}} for _ in range(2)]
    assert len(interventions)==2
    for context_id in range(2):
        for masked_sent_id in [1,2]:
            for rep_type in rep_types:
                if rep_type=='attention':
                    if args.test or args.intervention_type!='swap':
                        attn = GetReps(model,context_id,layer_id,head_id,'',rep_type,
                                    outputs[f'masked_sent_{masked_sent_id}_context_{context_id+1}'],
                                    token_ids[context_id][f'masked_sent_{masked_sent_id}'],args)
                    else:
                        attn = GetReps(model,context_id,layer_id,head_id,'',rep_type,
                                    outputs[f'masked_sent_{masked_sent_id}_context_{2-context_id}'],
                                    token_ids[1-context_id][f'masked_sent_{masked_sent_id}'],args)
                    assert f'attention_{layer_id}_{head_id}' not in interventions[context_id][f'masked_sent_{masked_sent_id}']
                    interventions[context_id][f'masked_sent_{masked_sent_id}'][f'attention_{layer_id}_{head_id}'] = attn
                elif rep_type=='q_and_k':
                    assert args.intervention_type=='swap'
                    assert len(pos_types)%2==0
                    for pos_pair_id in range(len(pos_types)//2):
                        k_pos = token_ids[context_id][f'masked_sent_{masked_sent_id}'][f'{pos_types[2*pos_pair_id]}']
                        q_pos = token_ids[context_id][f'masked_sent_{masked_sent_id}'][f'{pos_types[2*pos_pair_id+1]}']
                        if args.test:
                            k_vec = GetReps(model,context_id,layer_id,head_id,pos_types[2*pos_pair_id],'key',
                                        outputs[f'masked_sent_{masked_sent_id}_context_{context_id+1}'],
                                        token_ids[context_id][f'masked_sent_{masked_sent_id}'],args)
                            q_vec = GetReps(model,context_id,layer_id,head_id,pos_types[2*pos_pair_id+1],'query',
                                        outputs[f'masked_sent_{masked_sent_id}_context_{context_id+1}'],
                                        token_ids[context_id][f'masked_sent_{masked_sent_id}'],args)
                        else:
                            k_vec = GetReps(model,context_id,layer_id,head_id,pos_types[2*pos_pair_id],'key',
                                        outputs[f'masked_sent_{masked_sent_id}_context_{2-context_id}'],
                                        token_ids[1-context_id][f'masked_sent_{masked_sent_id}'],args)
                            q_vec = GetReps(model,context_id,layer_id,head_id,pos_types[2*pos_pair_id+1],'query',
                                        outputs[f'masked_sent_{masked_sent_id}_context_{2-context_id}'],
                                        token_ids[1-context_id][f'masked_sent_{masked_sent_id}'],args)
                        #if pos_types[2*pos_pair_id]!='context' or not args.no_eq_len_condition:
                        assert len(k_pos)==len(k_vec)
                        #if pos_types[2*pos_pair_id+1]!='context' or not args.no_eq_len_condition:
                        assert len(q_pos)==len(q_vec)
                        if f'key_{layer_id}_{head_id}' not in interventions[context_id][f'masked_sent_{masked_sent_id}']:
                            interventions[context_id][f'masked_sent_{masked_sent_id}'][f'key_{layer_id}_{head_id}'] = []
                        interventions[context_id][f'masked_sent_{masked_sent_id}'][f'key_{layer_id}_{head_id}'].extend([(k_pos,k_vec)])
                        if f'query_{layer_id}_{head_id}' not in interventions[context_id][f'masked_sent_{masked_sent_id}']:
                            interventions[context_id][f'masked_sent_{masked_sent_id}'][f'query_{layer_id}_{head_id}'] = []
                        interventions[context_id][f'masked_sent_{masked_sent_id}'][f'query_{layer_id}_{head_id}'].extend([(q_pos,q_vec)])
                else:
                    assert args.intervention_type=='swap'
                    for pos_type in pos_types:
                        pos = token_ids[context_id][f'masked_sent_{masked_sent_id}'][f'{pos_type}']
                        if args.test:
                            vec = GetReps(model,context_id,layer_id,head_id,pos_type,rep_type,
                                        outputs[f'masked_sent_{masked_sent_id}_context_{context_id+1}'],
                                        token_ids[context_id][f'masked_sent_{masked_sent_id}'],args)
                        else:
                            vec = GetReps(model,context_id,layer_id,head_id,pos_type,rep_type,
                                        outputs[f'masked_sent_{masked_sent_id}_context_{2-context_id}'],
                                        token_ids[1-context_id][f'masked_sent_{masked_sent_id}'],args)
                        #if pos_type!='context' or not args.no_eq_len_condition:
                        assert len(pos)==len(vec)
                        if f'{rep_type}_{layer_id}_{head_id}' not in interventions[context_id][f'masked_sent_{masked_sent_id}']:
                            interventions[context_id][f'masked_sent_{masked_sent_id}'][f'{rep_type}_{layer_id}_{head_id}'] = []
                        interventions[context_id][f'masked_sent_{masked_sent_id}'][f'{rep_type}_{layer_id}_{head_id}'].extend([(pos,vec)])
    if verbose:
        for intervention in [interventions[0]['masked_sent_1'],interventions[0]['masked_sent_2'],interventions[1]['masked_sent_1'],interventions[1]['masked_sent_2']]:
            for key,value in intervention.items():
                print(key)
                for pair in value:
                    print(pair[0],pair[1].shape)
    return interventions


def ApplyInterventionsLayer(interventions,model,pos_types,rep_types,outputs,token_ids,option_tokens_lists,args):
    for masked_sent_id in [1,2]:
        for context_id in [1,2]:
            assert len(outputs[f'masked_sent_{masked_sent_id}_context_{context_id}'][1][0].shape)==3
    int_outputs_1_context_1 = skeleton_model(0,outputs['masked_sent_1_context_1'][1][0].expand(model.config.num_attention_heads,-1,-1),
                                            model,interventions[0]['masked_sent_1'],args)
    int_outputs_2_context_1 = skeleton_model(0,outputs['masked_sent_2_context_1'][1][0].expand(model.config.num_attention_heads,-1,-1),
                                            model,interventions[0]['masked_sent_2'],args)
    int_outputs_1_context_2 = skeleton_model(0,outputs['masked_sent_1_context_2'][1][0].expand(model.config.num_attention_heads,-1,-1),
                                            model,interventions[1]['masked_sent_1'],args)
    int_outputs_2_context_2 = skeleton_model(0,outputs['masked_sent_2_context_2'][1][0].expand(model.config.num_attention_heads,-1,-1),
                                            model,interventions[1]['masked_sent_2'],args)

    for sent_id in [1,2]:
        for feature in ['option_1','option_2','context','masks','other','period','cls','sep']:
            assert torch.all(token_ids[0][f'masked_sent_{sent_id}'][feature]==token_ids[1][f'masked_sent_{sent_id}'][feature])

    token_ids_new_1 = token_ids[0]
    token_ids_new_2 = token_ids[1]

    option_tokens_list_1 = option_tokens_lists[0]
    option_tokens_list_2 = option_tokens_lists[1]

    choice_probs_sum_1,choice_probs_ave_1 = EvaluatePredictions(int_outputs_1_context_1[0],int_outputs_2_context_1[0],
                                                                token_ids_new_1,option_tokens_list_1,args)
    choice_probs_sum_2,choice_probs_ave_2 = EvaluatePredictions(int_outputs_1_context_2[0],int_outputs_2_context_2[0],
                                                                token_ids_new_2,option_tokens_list_2,args)

    attn_1_context_1 = EvaluateAttention(convert_to_numpy(int_outputs_1_context_1[2]),
                                        token_ids_new_1['masked_sent_1'],prediction_task=True,last_only=True)
    attn_2_context_2 = EvaluateAttention(convert_to_numpy(int_outputs_2_context_2[2]),
                                        token_ids_new_2['masked_sent_2'],prediction_task=True,last_only=True)

    results = {}
    results['sum_1'] = choice_probs_sum_1
    results['sum_2'] = choice_probs_sum_2
    results['ave_1'] = choice_probs_ave_1
    results['ave_2'] = choice_probs_ave_2
    results['masks-option-diff_1'] = attn_1_context_1['masks-option_1']-attn_1_context_1['masks-option_2']
    results['masks-option-diff_2'] = attn_2_context_2['masks-option_1']-attn_2_context_2['masks-option_2']
    return results

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
        option_tokens_lists = [option_tokens_list_1, option_tokens_list_2]

        out_dict = {}
        choice_probs_sum_1, choice_probs_ave_1 = EvaluatePredictions(outputs_1[0][0],outputs_1[1][0],token_ids_1,option_tokens_list_1,args)
        choice_probs_sum_2, choice_probs_ave_2 = EvaluatePredictions(outputs_2[0][0],outputs_2[1][0],token_ids_2,option_tokens_list_2,args)

        attn_1_context_1 = EvaluateAttention(convert_to_numpy(outputs_1[0][2]),
                                            token_ids_1['masked_sent_1'],prediction_task=True,last_only=True)
        attn_2_context_2 = EvaluateAttention(convert_to_numpy(outputs_2[1][2]),
                                            token_ids_2['masked_sent_2'],prediction_task=True,last_only=True)
        out_dict['original'] = {}
        out_dict['original']['sum_1'] = choice_probs_sum_1
        out_dict['original']['sum_2'] = choice_probs_sum_2
        out_dict['original']['ave_1'] = choice_probs_ave_1
        out_dict['original']['ave_2'] = choice_probs_ave_2
        out_dict['original']['masks-option-diff_1'] = attn_1_context_1['masks-option_1']-attn_1_context_1['masks-option_2']
        out_dict['original']['masks-option-diff_2'] = attn_2_context_2['masks-option_1']-attn_2_context_2['masks-option_2']

        for layer_id in range(model.config.num_hidden_layers):
            if str(layer_id) in args.layer.split('-') or args.layer=='all':
                interventions = []
                for head_id in range(model.config.num_attention_heads):
                    if str(head_id) in args.head.split('-') or args.head=='all':
                        interventions = CreateInterventions(model,interventions,layer_id,head_id,pos_types,rep_types,
                                                            outputs,token_ids,args)
                        if args.cascade:
                            # include interventions for preceding layers
                            for pre_layer_id in range(layer_id):
                                if str(pre_layer_id) in args.layer.split('-') or args.layer=='all':
                                    interventions = CreateInterventions(model,interventions,pre_layer_id,head_id,pos_types,rep_types,
                                                                        outputs,token_ids,args)

                assert f'layer_{layer_id}' not in out_dict
                out_dict[f'layer_{layer_id}'] = ApplyInterventionsLayer(interventions,model,pos_types,rep_types,
                                                                        outputs,token_ids,option_tokens_lists,args)
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
    parser.add_argument('--layer', type = str, default='all')
    parser.add_argument('--head', type = str, default='all')
    parser.add_argument('--intervention_type',type=str,default='swap',
                        choices=['swap','correct_option_attn','incorrect_option_attn',
                                'context_attn','other_attn',
                                'option_context_attn','option_masks_attn',
                                'context_context_attn','context_masks_attn',
                                'lesion_context_attn','lesion_attn',
                                'scramble_masks_attn'])
    parser.add_argument('--test',dest='test',action='store_true')
    parser.add_argument('--no_eq_len_condition',dest='no_eq_len_condition',action='store_true')
    parser.add_argument('--cascade',dest='cascade',action='store_true')
    parser.add_argument('--multihead',dest='multihead',action='store_true')
    parser.set_defaults(test=False,no_eq_len_condition=False,cascade=False,multihead=False)
    args = parser.parse_args()
    print(f'running with {args}')

    if 'attention' in args.rep_type.split('-'):
        assert not args.no_eq_len_condition

    if args.pos_type is None:
        assert args.rep_type=='attention'

    if args.test:
        test_id = '_test'
    else:
        test_id = ''
    if args.cascade:
        cascade_id = '_cascade'
    else:
        cascade_id = ''
    if args.multihead:
        multihead_id = '_multihead'
    else:
        multihead_id = ''

    head,text = LoadDataset(args)
    model, tokenizer, mask_id, args = LoadModel(args)

    if args.pos_type is None:
        if args.dataset=='superglue':
            out_file_name = f'{os.environ.get("DATA_PATH")}/superglue_wsc_intervention_{args.intervention_type}'\
                            +f'_{args.rep_type}_{args.model}_{args.stimuli}'\
                            +f'_layer_{args.layer}_head_{args.head}{cascade_id}{multihead_id}{test_id}'
        elif args.dataset=='winogrande':
            out_file_name = f'{os.environ.get("DATA_PATH")}/winogrande_{args.size}_intervention_{args.intervention_type}'\
                            +f'_{args.rep_type}_{args.model}'\
                            +f'_layer_{args.layer}_head_{args.head}{cascade_id}{multihead_id}{test_id}'
    else:
        if args.dataset=='superglue':
            out_file_name = f'{os.environ.get("DATA_PATH")}/superglue_wsc_intervention_{args.intervention_type}'\
                            +f'_{args.pos_type}_{args.rep_type}_{args.model}_{args.stimuli}'\
                            +f'_layer_{args.layer}_head_{args.head}{cascade_id}{multihead_id}{test_id}'
        elif args.dataset=='winogrande':
            out_file_name = f'{os.environ.get("DATA_PATH")}/winogrande_{args.size}_intervention_{args.intervention_type}'\
                            +f'_{args.pos_type}_{args.rep_type}_{args.model}'\
                            +f'_layer_{args.layer}_head_{args.head}{cascade_id}{multihead_id}{test_id}'

    with open(f'{out_file_name}.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(head+['interv_type','layer_id','head_id','original_score','score','effect_ave',
                                *[f'masks-option-diff_{head_id}' for head_id in range(model.config.num_attention_heads)],
                                *[f'masks-option-diff_effect_{head_id}' for head_id in range(model.config.num_attention_heads)]])
        sent_num = 0
        for line in text[:500]:
            if args.pos_type is None:
                results = ApplyInterventions(head,line,[],args.rep_type.split('-'),model,tokenizer,mask_id,args)
            else:
                results = ApplyInterventions(head,line,args.pos_type.split('-'),args.rep_type.split('-'),model,tokenizer,mask_id,args)
            if type(results) is str:
                continue
            else:
                sent_num += 1
                original_1 = results['original']['ave_1'][0]-results['original']['ave_1'][1]
                original_2 = results['original']['ave_2'][0]-results['original']['ave_2'][1]
                original_attn_1 = np.array([results['original']['masks-option-diff_1'][target_head_id]
                                            for target_head_id in range(model.config.num_attention_heads)])
                original_attn_2 = np.array([results['original']['masks-option-diff_2'][target_head_id]
                                            for target_head_id in range(model.config.num_attention_heads)])
                original_score = (original_1>0)&(original_2<0)
                for layer_id in range(model.config.num_hidden_layers):
                    result_dict = results[f'layer_{layer_id}']
                    for head_id in range(model.config.num_attention_heads):
                        interv_1 = result_dict['ave_1'][0][head_id]-result_dict['ave_1'][1][head_id]
                        interv_2 = result_dict['ave_2'][0][head_id]-result_dict['ave_2'][1][head_id]
                        interv_attn_1 = np.array([result_dict['masks-option-diff_1'][head_id,target_head_id]
                                                    for target_head_id in range(model.config.num_attention_heads)])
                        interv_attn_2 = np.array([result_dict['masks-option-diff_2'][head_id,target_head_id]
                                                    for target_head_id in range(model.config.num_attention_heads)])

                        score = (interv_1>0)&(interv_2<0)
                        effect_1 = original_1-interv_1
                        effect_2 = interv_2-original_2
                        effect_attn_1 = original_attn_1-interv_attn_1
                        effect_attn_2 = interv_attn_2-original_attn_2
                        writer.writerow(line+['interv',layer_id,head_id,original_score,score,(effect_1+effect_2)/2,
                                                *list((interv_attn_1-interv_attn_2)/2),
                                                *list((effect_attn_1+effect_attn_2)/2)])
                        writer.writerow(line+['original',layer_id,head_id,original_score,original_score,0.0,
                                                *list((original_attn_1-original_attn_2)/2),
                                                *[0.0 for _ in range(model.config.num_attention_heads)]])

    print(f'Time it took: {time.time()-start}')
    print(f'# sentences processed: {sent_num}\n')
