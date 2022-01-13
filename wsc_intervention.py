import numpy as np
import torch
import pickle
import torch.nn.functional as F
import argparse
import json
import csv
from wsc_utils import CalcOutputs, EvaluatePredictions, LoadDataset, LoadModel, GetReps
from model_skeleton import skeleton_model, ExtractAttnLayer
import pandas as pd
import time
import math

def FixAttn(mat,token_ids,in_pos,out_pos,args,reverse=False):
    if not args.test:
        if reverse:
            patch = torch.ones((len(token_ids[out_pos]),mat.shape[1])).to(args.device)/(mat.shape[1]-len(token_ids[in_pos]))
            patch[:,token_ids[in_pos]] = 0
        else:
            patch = torch.zeros((len(token_ids[out_pos]),mat.shape[1])).to(args.device)
            patch[:,token_ids[in_pos]] = 1/len(token_ids[in_pos])
        mat[token_ids[out_pos],:] = patch.clone()
    return mat

def ScrambleAttn(mat,token_ids,out_pos,args):
    if not args.test:
        rand_ids = np.random.permutation(mat.shape[1])
        patch = torch.tensor([[mat[out_pos_id][rand_id] for rand_id in rand_ids]
                            for out_pos_id in token_ids[out_pos]]).to(args.device)
        mat[token_ids[out_pos],:] = patch.clone()
    return mat

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
                    assert len(pos_types)==2
                    k_pos = token_ids[context_id][f'masked_sent_{masked_sent_id}'][f'{pos_types[0]}']
                    q_pos = token_ids[context_id][f'masked_sent_{masked_sent_id}'][f'{pos_types[1]}']
                    if args.test:
                        k_vec = GetReps(model,context_id,layer_id,head_id,pos_types[0],'key',
                                    outputs[f'masked_sent_{masked_sent_id}_context_{context_id+1}'],
                                    token_ids[context_id][f'masked_sent_{masked_sent_id}'],args)
                        q_vec = GetReps(model,context_id,layer_id,head_id,pos_types[1],'query',
                                    outputs[f'masked_sent_{masked_sent_id}_context_{context_id+1}'],
                                    token_ids[context_id][f'masked_sent_{masked_sent_id}'],args)
                    else:
                        k_vec = GetReps(model,context_id,layer_id,head_id,pos_types[0],'key',
                                    outputs[f'masked_sent_{masked_sent_id}_context_{2-context_id}'],
                                    token_ids[1-context_id][f'masked_sent_{masked_sent_id}'],args)
                        q_vec = GetReps(model,context_id,layer_id,head_id,pos_types[1],'query',
                                    outputs[f'masked_sent_{masked_sent_id}_context_{2-context_id}'],
                                    token_ids[1-context_id][f'masked_sent_{masked_sent_id}'],args)
                    if pos_types[0]!='context' or not args.no_eq_len_condition:
                        assert len(k_pos)==len(k_vec)
                    if pos_types[1]!='context' or not args.no_eq_len_condition:
                        assert len(q_pos)==len(q_vec)
                    if f'key_{layer_id}' not in interventions[context_id][f'masked_sent_{masked_sent_id}']:
                        interventions[context_id][f'masked_sent_{masked_sent_id}'][f'key_{layer_id}'] = []
                    interventions[context_id][f'masked_sent_{masked_sent_id}'][f'key_{layer_id}'].extend([(k_pos,k_vec)])
                    if f'query_{layer_id}' not in interventions[context_id][f'masked_sent_{masked_sent_id}']:
                        interventions[context_id][f'masked_sent_{masked_sent_id}'][f'query_{layer_id}'] = []
                    interventions[context_id][f'masked_sent_{masked_sent_id}'][f'query_{layer_id}'].extend([(q_pos,q_vec)])
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
                        if pos_type!='context' or not args.no_eq_len_condition:
                            assert len(pos)==len(vec)
                        if f'{rep_type}_{layer_id}' not in interventions[context_id][f'masked_sent_{masked_sent_id}']:
                            interventions[context_id][f'masked_sent_{masked_sent_id}'][f'{rep_type}_{layer_id}'] = []
                        interventions[context_id][f'masked_sent_{masked_sent_id}'][f'{rep_type}_{layer_id}'].extend([(pos,vec)])
    if verbose:
        for intervention in [interventions[0]['masked_sent_1'],interventions[0]['masked_sent_2'],interventions[1]['masked_sent_1'],interventions[1]['masked_sent_2']]:
            for key,value in intervention.items():
                print(key)
                for pair in value:
                    print(pair[0],pair[1].shape)
    return interventions


def ApplyInterventionsLayer(interventions,model,layer_id,pos_types,rep_types,outputs,token_ids,option_tokens_lists,args):
    int_logits_1_context_1 = skeleton_model(layer_id,outputs['masked_sent_1_context_1'][1][layer_id],
                                            model,interventions[0]['masked_sent_1'],args)
    int_logits_2_context_1 = skeleton_model(layer_id,outputs['masked_sent_2_context_1'][1][layer_id],
                                            model,interventions[0]['masked_sent_2'],args)
    int_logits_1_context_2 = skeleton_model(layer_id,outputs['masked_sent_1_context_2'][1][layer_id],
                                            model,interventions[1]['masked_sent_1'],args)
    int_logits_2_context_2 = skeleton_model(layer_id,outputs['masked_sent_2_context_2'][1][layer_id],
                                            model,interventions[1]['masked_sent_2'],args)

    if 'context' in pos_types and not args.test:
        token_ids_new_1 = token_ids[1]
        token_ids_new_2 = token_ids[0]
    else:
        token_ids_new_1 = token_ids[0]
        token_ids_new_2 = token_ids[1]

    option_tokens_list_1 = option_tokens_lists[0]
    option_tokens_list_2 = option_tokens_lists[1]

    choice_probs_sum_1,choice_probs_ave_1 = EvaluatePredictions(int_logits_1_context_1,int_logits_2_context_1,
                                                                token_ids_new_1,option_tokens_list_1,args)
    choice_probs_sum_2,choice_probs_ave_2 = EvaluatePredictions(int_logits_1_context_2,int_logits_2_context_2,
                                                                token_ids_new_2,option_tokens_list_2,args)

    results = {}
    results['sum_1'] = choice_probs_sum_1
    results['sum_2'] = choice_probs_sum_2
    results['ave_1'] = choice_probs_ave_1
    results['ave_2'] = choice_probs_ave_2
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
        out_dict['original'] = {}
        out_dict['original']['sum_1'] = choice_probs_sum_1
        out_dict['original']['sum_2'] = choice_probs_sum_2
        out_dict['original']['ave_1'] = choice_probs_ave_1
        out_dict['original']['ave_2'] = choice_probs_ave_2

        for layer_id in range(model.config.num_hidden_layers):
            if str(layer_id) in args.layer.split('-') or args.layer=='all':
                if 'attention' in rep_types and args.multihead:
                    assert not args.no_eq_len_condition
                    interventions = []
                    for head_id in range(model.config.num_attention_heads):
                        if str(head_id) in args.head.split('-') or args.head=='all':
                            interventions = CreateInterventions(model,interventions,layer_id,head_id,pos_types,rep_types,
                                                                outputs,token_ids,args)
                            if args.cascade:
                                # include interventions for later layers
                                for pre_layer_id in range(layer_id):
                                    if str(pre_layer_id) in args.layer.split('-') or args.layer=='all':
                                        interventions = CreateInterventions(model,interventions,pre_layer_id,head_id,pos_types,rep_types,
                                                                            outputs,token_ids,args)
                    assert f'layer_{layer_id}' not in out_dict
                    if args.cascade:
                        out_dict[f'layer_{layer_id}'] = ApplyInterventionsLayer(interventions,model,0,pos_types,rep_types,
                                                                                outputs,token_ids,option_tokens_lists,args)
                    else:
                        out_dict[f'layer_{layer_id}'] = ApplyInterventionsLayer(interventions,model,layer_id,pos_types,rep_types,
                                                                                outputs,token_ids,option_tokens_lists,args)
                elif 'attention' in rep_types and not args.multihead:
                    assert not args.no_eq_len_condition
                    for head_id in range(model.config.num_attention_heads):
                        if str(head_id) in args.head.split('-') or args.head=='all':
                            interventions = []
                            interventions = CreateInterventions(model,interventions,layer_id,head_id,pos_types,rep_types,
                                                                outputs,token_ids,args)
                            if args.cascade:
                                # include interventions for later layers
                                for pre_layer_id in range(layer_id):
                                    if str(pre_layer_id) in args.layer.split('-') or args.layer=='all':
                                        interventions = CreateInterventions(model,interventions,pre_layer_id,head_id,pos_types,rep_types,
                                                                            outputs,token_ids,args)
                            assert f'layer_{layer_id}_{head_id}' not in out_dict
                            if args.cascade:
                                out_dict[f'layer_{layer_id}_{head_id}'] = ApplyInterventionsLayer(interventions,model,0,pos_types,rep_types,
                                                                                                    outputs,token_ids,option_tokens_lists,args)
                            else:
                                out_dict[f'layer_{layer_id}_{head_id}'] = ApplyInterventionsLayer(interventions,model,layer_id,pos_types,rep_types,
                                                                                                    outputs,token_ids,option_tokens_lists,args)
                else:
                    interventions = []
                    interventions = CreateInterventions(model,interventions,layer_id,None,pos_types,rep_types,
                                                        outputs,token_ids,args)
                    if args.cascade:
                        # include interventions for later layers
                        for pre_layer_id in range(layer_id):
                            if str(pre_layer_id) in args.layer.split('-') or args.layer=='all':
                                interventions = CreateInterventions(model,interventions,pre_layer_id,None,pos_types,rep_types,
                                                                    outputs,token_ids,args)
                    assert f'layer_{layer_id}' not in out_dict
                    if args.cascade:
                        out_dict[f'layer_{layer_id}'] = ApplyInterventionsLayer(interventions,model,0,pos_types,rep_types,
                                                                                outputs,token_ids,option_tokens_lists,args)
                    else:
                        out_dict[f'layer_{layer_id}'] = ApplyInterventionsLayer(interventions,model,layer_id,pos_types,rep_types,
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
    # context and masks perturbations should be the last since they may change the sentence length
    #assert args.rep_type in ['layer','query','key','value','layer-query','key-value','layer-query-key-value']
    #assert args.pos_type in ['option_1','option_2','option_1-option_2','context','masks','period','cls','sep','cls-sep','cls-period-sep','option_1-option_2-context','other']
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
            out_file_name = f'datafile/superglue_wsc_intervention_{args.intervention_type}'\
                            +f'_{args.rep_type}_{args.model}_{args.stimuli}'\
                            +f'_layer_{args.layer}_head_{args.head}{cascade_id}{multihead_id}{test_id}.pkl'
        elif args.dataset=='winogrande':
            out_file_name = f'datafile/winogrande_{args.size}_intervention_{args.intervention_type}'\
                            +f'_{args.rep_type}_{args.model}'\
                            +f'_layer_{args.layer}_head_{args.head}{cascade_id}{multihead_id}{test_id}.pkl'
    else:
        if args.dataset=='superglue':
            out_file_name = f'datafile/superglue_wsc_intervention_{args.intervention_type}'\
                            +f'_{args.pos_type}_{args.rep_type}_{args.model}_{args.stimuli}'\
                            +f'_layer_{args.layer}_head_{args.head}{cascade_id}{multihead_id}{test_id}.pkl'
        elif args.dataset=='winogrande':
            out_file_name = f'datafile/winogrande_{args.size}_intervention_{args.intervention_type}'\
                            +f'_{args.pos_type}_{args.rep_type}_{args.model}'\
                            +f'_layer_{args.layer}_head_{args.head}{cascade_id}{multihead_id}{test_id}.pkl'
    with open(out_file_name,'wb') as f:
        pickle.dump(out_dict,f)

    print(f'Time it took: {time.time()-start}')
    print(f'# sentences processed: {len(list(out_dict.keys()))}\n')
