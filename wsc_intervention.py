# export MY_DATA_PATH='YOUR PATH TO DATA FILES'
import numpy as np
import torch
import pickle
import torch.nn.functional as F
import argparse
import json
import csv
from wsc_utils import CalcOutputs, EvaluatePredictions, LoadDataset, LoadModel, GetReps, ExtractQKV, EvaluateQKV
from model_skeleton import skeleton_model, ExtractAttnLayer
from wsc_attention import EvaluateAttention,convert_to_numpy
import pandas as pd
import time
import math
import os

def CreateInterventionsAttention(interventions,outputs,token_ids,layer_id,head_id,condition_id,pair_condition_id,context_id,args):
    if args.test or args.intervention_type!='swap':
        attn = GetReps(outputs[condition_id],token_ids[condition_id],
                        layer_id,head_id,'','attention',args,context_id=context_id)
    else:
        attn = GetReps(outputs[pair_condition_id],token_ids[pair_condition_id],
                        layer_id,head_id,'','attention',args)
    assert f'attention_{layer_id}_{head_id}' not in interventions[condition_id]
    interventions[condition_id][f'attention_{layer_id}_{head_id}'] = attn
    return interventions

def CreateInterventionsQandK(interventions,outputs,token_ids,layer_id,head_id,pos_types,condition_id,pair_condition_id,args):
    assert args.intervention_type=='swap'
    assert len(pos_types)%2==0
    for pos_pair_id in range(len(pos_types)//2):
        k_pos = token_ids[condition_id][f'{pos_types[2*pos_pair_id]}']
        q_pos = token_ids[condition_id][f'{pos_types[2*pos_pair_id+1]}']
        if args.test:
            k_vec = GetReps(outputs[condition_id],token_ids[condition_id],
                            layer_id,head_id,pos_types[2*pos_pair_id],'key',args)
            q_vec = GetReps(outputs[condition_id],token_ids[condition_id],
                            layer_id,head_id,pos_types[2*pos_pair_id+1],'query',args)
        else:
            k_vec = GetReps(outputs[pair_condition_id],token_ids[pair_condition_id],
                            layer_id,head_id,pos_types[2*pos_pair_id],'key',args)
            q_vec = GetReps(outputs[pair_condition_id],token_ids[pair_condition_id],
                            layer_id,head_id,pos_types[2*pos_pair_id+1],'query',args)
        #if pos_types[2*pos_pair_id]!='context' or not args.no_eq_len_condition:
        assert len(k_pos)==len(k_vec)
        #if pos_types[2*pos_pair_id+1]!='context' or not args.no_eq_len_condition:
        assert len(q_pos)==len(q_vec)
        if f'key_{layer_id}_{head_id}' not in interventions[condition_id]:
            interventions[condition_id][f'key_{layer_id}_{head_id}'] = []
        interventions[condition_id][f'key_{layer_id}_{head_id}'].extend([(k_pos,k_vec)])
        if f'query_{layer_id}_{head_id}' not in interventions[condition_id]:
            interventions[condition_id][f'query_{layer_id}_{head_id}'] = []
        interventions[condition_id][f'query_{layer_id}_{head_id}'].extend([(q_pos,q_vec)])
    return interventions

def CreateInterventionsLayers(interventions,outputs,token_ids,layer_id,head_id,pos_types,rep_type,condition_id,pair_condition_id,args):
    assert args.intervention_type=='swap'
    for pos_type in pos_types:
        pos = token_ids[condition_id][f'{pos_type}']
        if args.test:
            vec = GetReps(outputs[condition_id],token_ids[condition_id],
                        layer_id,head_id,pos_type,rep_type,args)
        else:
            vec = GetReps(outputs[pair_condition_id],token_ids[pair_condition_id],
                        layer_id,head_id,pos_type,rep_type,args)
        #if pos_type!='context' or not args.no_eq_len_condition:
        assert len(pos)==len(vec)
        if f'{rep_type}_{layer_id}_{head_id}' not in interventions[condition_id]:
            interventions[condition_id][f'{rep_type}_{layer_id}_{head_id}'] = []
        interventions[condition_id][f'{rep_type}_{layer_id}_{head_id}'].extend([(pos,vec)])
    return interventions

def CreateInterventions(interventions,layer_id,head_id,pos_types,rep_types,outputs,token_ids,args,verbose=False):
    if not bool(interventions):
        if args.stimuli=='original' or 'verb' in args.stimuli:
            interventions = {f'masked_sent_{masked_sent_id}_context_{context_id}':{}
                                for masked_sent_id in [1,2] for context_id in [1,2]}
        else:
            assert args.stimuli=='control_combined'
            interventions = {f'context_{context_id}':{} for context_id in [1,2]}
    for context_id in [1,2]:
        if args.stimuli=='original' or 'verb' in args.stimuli:
            for masked_sent_id in [1,2]:
                condition_id = f'masked_sent_{masked_sent_id}_context_{context_id}'
                pair_condition_id = f'masked_sent_{masked_sent_id}_context_{3-context_id}'
                for rep_type in rep_types:
                    if rep_type=='attention':
                        interventions = CreateInterventionsAttention(interventions,outputs,token_ids,layer_id,head_id,
                                                                        condition_id,pair_condition_id,context_id,args)
                    elif rep_type=='q_and_k':
                        interventions = CreateInterventionsQandK(interventions,outputs,token_ids,layer_id,head_id,pos_types,
                                                                    condition_id,pair_condition_id,args)
                    else:
                        interventions = CreateInterventionsLayers(interventions,outputs,token_ids,layer_id,head_id,pos_types,rep_type,
                                                                    condition_id,pair_condition_id,args)
                if verbose:
                    for key,value in interventions[condition_id].items():
                        print(key)
                        for pair in value:
                            print(pair[0],pair[1].shape)
        else:
            assert args.stimuli=='control_combined'
            condition_id = f'context_{context_id}'
            pair_condition_id = f'context_{3-context_id}'
            for rep_type in rep_types:
                if rep_type=='attention':
                    interventions = CreateInterventionsAttention(interventions,outputs,token_ids,layer_id,head_id,
                                                                    condition_id,pair_condition_id,context_id,args)
                elif rep_type=='q_and_k':
                    interventions = CreateInterventionsQandK(interventions,outputs,token_ids,layer_id,head_id,pos_types,
                                                                condition_id,pair_condition_id,args)
                else:
                    interventions = CreateInterventionsLayers(interventions,outputs,token_ids,layer_id,head_id,pos_types,rep_type,
                                                                condition_id,pair_condition_id,args)
            if verbose:
                for key,value in interventions[condition_id].items():
                    print(key)
                    for pair in value:
                        print(pair[0],pair[1].shape)
    return interventions


def ApplyInterventionsLayer(interventions,model,pos_types,rep_types,outputs,token_ids,option_tokens_lists,args):
    int_outputs = {}
    for context_id in [1,2]:
        if args.stimuli=='original' or 'verb' in args.stimuli:
            for masked_sent_id in [1,2]:
                condition_id = f'masked_sent_{masked_sent_id}_context_{context_id}'
                assert len(outputs[condition_id][1][0].shape)==3
                int_outputs[condition_id] = skeleton_model(0,outputs[condition_id][1][0],
                                                            model,interventions[condition_id],args)
                pair_condition_id = f'masked_sent_{masked_sent_id}_context_{3-context_id}'
                for feature in ['options','option_1','option_2','context','masks','other','period','cls','sep']:
                    assert torch.all(token_ids[condition_id][feature]==token_ids[pair_condition_id][feature])
        else:
            assert args.stimuli=='control_combined'
            condition_id = f'context_{context_id}'
            assert len(outputs[condition_id][1][0].shape)==3
            int_outputs[condition_id] = skeleton_model(0,outputs[condition_id][1][0],
                                                        model,interventions[condition_id],args)
            pair_condition_id = f'context_{3-context_id}'
            for feature in ['options','option_1','option_2','context','masks','other','period','cls','sep']:
                assert torch.all(token_ids[condition_id][feature]==token_ids[pair_condition_id][feature])

    return OutputResults(int_outputs,token_ids,option_tokens_lists,args)

def OutputResults(outputs,token_ids,option_tokens_lists,args):
    out_dict = {}
    if args.stimuli=='original' or 'verb' in args.stimuli:
        assert token_ids['masked_sent_1_context_1']['pron_id']==token_ids['masked_sent_2_context_1']['pron_id']
        assert token_ids['masked_sent_1_context_2']['pron_id']==token_ids['masked_sent_2_context_2']['pron_id']
        pron_token_id_1 = token_ids['masked_sent_1_context_1']['pron_id']
        pron_token_id_2 = token_ids['masked_sent_2_context_2']['pron_id']
        choice_probs_sum_1, choice_probs_ave_1 = EvaluatePredictions(outputs['masked_sent_1_context_1'][0],
                                                                    outputs['masked_sent_2_context_1'][0],
                                                                    pron_token_id_1,option_tokens_lists[0],args)
        choice_probs_sum_2, choice_probs_ave_2 = EvaluatePredictions(outputs['masked_sent_1_context_2'][0],
                                                                    outputs['masked_sent_2_context_2'][0],
                                                                    pron_token_id_2,option_tokens_lists[1],args)
        out_dict['sum_1'] = choice_probs_sum_1
        out_dict['sum_2'] = choice_probs_sum_2
        out_dict['ave_1'] = choice_probs_ave_1
        out_dict['ave_2'] = choice_probs_ave_2

        for masked_sent_id in [1,2]:
            for context_id in [1,2]:
                condition_id = f'masked_sent_{masked_sent_id}_context_{context_id}'
                out_dict[f'qry_{condition_id}'] = ExtractQKV(outputs[condition_id][3][0][-1],
                                                            'masks',token_ids[condition_id])
                out_dict[f'key_{condition_id}'] = ExtractQKV(outputs[condition_id][3][1][-1],
                                                            'options',token_ids[condition_id])
        attn_1_context_1 = EvaluateAttention(convert_to_numpy(outputs['masked_sent_1_context_1'][2]),
                                            token_ids['masked_sent_1_context_1'],prediction_task=True,target_layer_id=-1)
        attn_2_context_2 = EvaluateAttention(convert_to_numpy(outputs['masked_sent_2_context_2'][2]),
                                            token_ids['masked_sent_2_context_2'],prediction_task=True,target_layer_id=-1)
        out_dict['masks-option-diff_1'] = attn_1_context_1['masks-option_1']-attn_1_context_1['masks-option_2']
        out_dict['masks-option-diff_2'] = attn_2_context_2['masks-option_1']-attn_2_context_2['masks-option_2']
    else:
        assert args.stimuli=='control_combined'
        for context_id in [1,2]:
            condition_id = f'context_{context_id}'
            out_dict[f'qry_{condition_id}'] = ExtractQKV(outputs[condition_id][3][0][-1],
                                                        'masks',token_ids[condition_id])
            out_dict[f'key_{condition_id}'] = ExtractQKV(outputs[condition_id][3][1][-1],
                                                        'options',token_ids[condition_id])

        attn_1_context_1 = EvaluateAttention(convert_to_numpy(outputs['context_1'][2]),
                                            token_ids['context_1'],target_layer_id=-1)
        attn_2_context_2 = EvaluateAttention(convert_to_numpy(outputs['context_2'][2]),
                                            token_ids['context_2'],target_layer_id=-1)

        out_dict['masks-option-diff_1'] = attn_1_context_1['pron_id-option_1']-attn_1_context_1['pron_id-option_2']
        out_dict['masks-option-diff_2'] = attn_2_context_2['pron_id-option_1']-attn_2_context_2['pron_id-option_2']

    return out_dict

def ApplyInterventions(head,line,pos_types,rep_types,model,tokenizer,mask_id,args):
    assert int(line[head.index('option_1_word_id_1')]) < int(line[head.index('option_2_word_id_1')])
    assert int(line[head.index('option_1_word_id_2')]) < int(line[head.index('option_2_word_id_2')])

    if args.stimuli=='original' or 'verb' in args.stimuli:
        outputs_1,token_ids_1,option_tokens_list_1,_ = CalcOutputs(head,line,1,model,tokenizer,mask_id,args,use_skeleton=True)
        outputs_2,token_ids_2,option_tokens_list_2,_ = CalcOutputs(head,line,2,model,tokenizer,mask_id,args,use_skeleton=True)
    else:
        assert args.stimuli=='control_combined'
        outputs_1,token_ids_1,option_tokens_list_1,_ = CalcOutputs(head,line,1,model,tokenizer,mask_id,args,use_skeleton=True,output_for_attn=True)
        outputs_2,token_ids_2,option_tokens_list_2,_ = CalcOutputs(head,line,2,model,tokenizer,mask_id,args,use_skeleton=True,output_for_attn=True)

    if CheckNumTokens(outputs_1,outputs_2,token_ids_1,token_ids_2,args) or args.no_eq_len_condition:
        outputs = {}
        token_ids = {}
        if args.stimuli=='original' or 'verb' in args.stimuli:
            for masked_sent_id in [1,2]:
                outputs[f'masked_sent_{masked_sent_id}_context_1'] = outputs_1[masked_sent_id-1]
                outputs[f'masked_sent_{masked_sent_id}_context_2'] = outputs_2[masked_sent_id-1]
                token_ids[f'masked_sent_{masked_sent_id}_context_1'] = token_ids_1[f'masked_sent_{masked_sent_id}']
                token_ids[f'masked_sent_{masked_sent_id}_context_2'] = token_ids_2[f'masked_sent_{masked_sent_id}']
                token_ids[f'masked_sent_{masked_sent_id}_context_1']['pron_id'] = token_ids_1['pron_id']
                token_ids[f'masked_sent_{masked_sent_id}_context_2']['pron_id'] = token_ids_2['pron_id']
        else:
            assert args.stimuli=='control_combined'
            outputs[f'context_1'] = outputs_1
            outputs[f'context_2'] = outputs_2
            token_ids[f'context_1'] = token_ids_1
            token_ids[f'context_2'] = token_ids_2

        option_tokens_lists = [option_tokens_list_1, option_tokens_list_2]

        out_dict = {}
        out_dict['original'] = OutputResults(outputs,token_ids,option_tokens_lists,args)

        for layer_id in range(model.config.num_hidden_layers):
            if str(layer_id) in args.layer.split('-') or args.layer=='all':
                interventions = []
                for head_id in range(model.config.num_attention_heads):
                    if str(head_id) in args.head.split('-') or args.head=='all':
                        interventions = CreateInterventions(interventions,layer_id,head_id,pos_types,rep_types,
                                                            outputs,token_ids,args)
                        if args.cascade:
                            # include interventions for preceding layers
                            for pre_layer_id in range(layer_id):
                                if str(pre_layer_id) in args.layer.split('-') or args.layer=='all':
                                    interventions = CreateInterventions(interventions,pre_layer_id,head_id,pos_types,rep_types,
                                                                        outputs,token_ids,args)

                assert f'layer_{layer_id}' not in out_dict
                out_dict[f'layer_{layer_id}'] = ApplyInterventionsLayer(interventions,model,pos_types,rep_types,
                                                                        outputs,token_ids,option_tokens_lists,args)
        return out_dict
    else:
        return 'number of tokens did not match'

def CheckNumTokens(outputs_1,outputs_2,token_ids_1,token_ids_2,args):
    features = ['option_1','option_2','context','masks','other','period','cls','sep']
    if 'verb' in args.stimuli:
        features.append(['verb'])
    if outputs_1[0][0].shape[1]==outputs_2[0][0].shape[1] and outputs_1[1][0].shape[1]==outputs_2[1][0].shape[1]:
        if args.stimuli=='original' or 'verb' in args.stimuli:
            for sent_id in [1,2]:
                for feature in features:
                    assert torch.all(token_ids_1[f'masked_sent_{sent_id}'][feature]==token_ids_2[f'masked_sent_{sent_id}'][feature])
        else:
            assert args.stimuli=='control_combined'
            for feature in features:
                assert torch.all(token_ids_1[feature]==token_ids_2[feature])
        return True
    else:
        return False

def EvaluateResults(result,head_id,args):
    if args.stimuli=='original' or 'verb' in args.stimuli:
        llr_1 = result['ave_1'][0][head_id]-result['ave_1'][1][head_id]
        llr_2 = result['ave_2'][0][head_id]-result['ave_2'][1][head_id]
    else:
        assert args.stimuli=='control_combined'
        llr_1, llr_2 = 0.0, 0.0
    attn_1 = np.array([result['masks-option-diff_1'][head_id,target_head_id]
                        for target_head_id in range(args.num_heads)])
    attn_2 = np.array([result['masks-option-diff_2'][head_id,target_head_id]
                        for target_head_id in range(args.num_heads)])
    score = (llr_1>0)&(llr_2<0)
    return llr_1,llr_2,attn_1,attn_2,score

if __name__=='__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--dataset', type = str, required = True, choices=['superglue','winogrande'])
    parser.add_argument('--stimuli', type = str,
                        choices=['original','control_gender','control_number',
                                'control_combined','original_verb','control_combined_verb'],
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
    parser.add_argument('--no_mask',dest='no_mask',action='store_true')
    parser.set_defaults(test=False,no_eq_len_condition=False,cascade=False,multihead=False,no_mask=False)
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
    args.num_layers = model.config.num_hidden_layers
    attn_layer = ExtractAttnLayer(0,model,args)
    args.num_heads = attn_layer.num_attention_heads
    args.head_dim = attn_layer.attention_head_size

    if args.pos_type is None:
        if args.dataset=='superglue':
            out_file_name = f'{os.environ.get("MY_DATA_PATH")}/superglue_wsc_intervention_{args.intervention_type}'\
                            +f'_{args.rep_type}_{args.model}_{args.stimuli}'\
                            +f'_layer_{args.layer}_head_{args.head}{cascade_id}{multihead_id}{test_id}'
        elif args.dataset=='winogrande':
            out_file_name = f'{os.environ.get("MY_DATA_PATH")}/winogrande_{args.size}_intervention_{args.intervention_type}'\
                            +f'_{args.rep_type}_{args.model}'\
                            +f'_layer_{args.layer}_head_{args.head}{cascade_id}{multihead_id}{test_id}'
    else:
        if args.dataset=='superglue':
            out_file_name = f'{os.environ.get("MY_DATA_PATH")}/superglue_wsc_intervention_{args.intervention_type}'\
                            +f'_{args.pos_type}_{args.rep_type}_{args.model}_{args.stimuli}'\
                            +f'_layer_{args.layer}_head_{args.head}{cascade_id}{multihead_id}{test_id}'
        elif args.dataset=='winogrande':
            out_file_name = f'{os.environ.get("MY_DATA_PATH")}/winogrande_{args.size}_intervention_{args.intervention_type}'\
                            +f'_{args.pos_type}_{args.rep_type}_{args.model}'\
                            +f'_layer_{args.layer}_head_{args.head}{cascade_id}{multihead_id}{test_id}'

    with open(f'{out_file_name}.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(head+['interv_type','layer_id','head_id','original_score','score','effect_ave',
                                *[f'masks-option-diff_{head_id}' for head_id in range(args.num_heads)],
                                *[f'masks-option-diff_effect_{head_id}' for head_id in range(args.num_heads)],
                                *[f'masks-qry-dist_effect_{head_id}' for head_id in range(args.num_heads)],
                                *[f'masks-qry-cos_effect_{head_id}' for head_id in range(args.num_heads)],
                                *[f'options-key-dist_effect_{head_id}' for head_id in range(args.num_heads)],
                                *[f'options-key-cos_effect_{head_id}' for head_id in range(args.num_heads)]])
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
                original_1,original_2,original_attn_1,original_attn_2,original_score = EvaluateResults(results['original'],0,args)

                original_qry_dist,original_qry_cos = EvaluateQKV('qry',results['original'],results['original'],0,0,args)
                original_key_dist,original_key_cos = EvaluateQKV('key',results['original'],results['original'],0,0,args)
                for layer_id in range(args.num_layers):
                    result_dict = results[f'layer_{layer_id}']
                    for head_id in range(args.num_heads):
                        interv_1,interv_2,interv_attn_1,interv_attn_2,interv_score = EvaluateResults(result_dict,head_id,args)

                        effect_1 = original_1-interv_1
                        effect_2 = interv_2-original_2
                        effect_attn_1 = original_attn_1-interv_attn_1
                        effect_attn_2 = interv_attn_2-original_attn_2

                        interv_qry_dist,interv_qry_cos = EvaluateQKV('qry',result_dict,results['original'],head_id,0,args)
                        interv_key_dist,interv_key_cos = EvaluateQKV('key',result_dict,results['original'],head_id,0,args)
                        assert len(interv_qry_dist)==args.num_heads and len(interv_qry_cos)==args.num_heads
                        assert len(interv_key_dist)==args.num_heads and len(interv_key_cos)==args.num_heads

                        writer.writerow(line+['interv',layer_id,head_id,original_score,interv_score,(effect_1+effect_2)/2,
                                                *list((interv_attn_1-interv_attn_2)/2),
                                                *list((effect_attn_1+effect_attn_2)/2),
                                                *list(original_qry_dist-interv_qry_dist),
                                                *list(interv_qry_cos-original_qry_cos),
                                                *list(original_key_dist-interv_key_dist),
                                                *list(interv_key_cos-original_key_cos)])
                        writer.writerow(line+['original',layer_id,head_id,original_score,original_score,0.0,
                                                *list((original_attn_1-original_attn_2)/2),
                                                *[0.0 for _ in range(args.num_heads)],
                                                *[0.0 for _ in range(args.num_heads)],
                                                *[0.0 for _ in range(args.num_heads)],
                                                *[0.0 for _ in range(args.num_heads)],
                                                *[0.0 for _ in range(args.num_heads)]])

    print(f'Time it took: {time.time()-start}')
    print(f'# sentences processed: {sent_num}\n')
