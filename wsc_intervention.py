import numpy as np
import torch
import pickle
import torch.nn.functional as F
import argparse
import json
import csv
from wsc_utils import CalcOutputs, EvaluatePredictions, LoadDataset, LoadModel
from model_skeleton import skeleton_model, swap_vecs_test
import pandas as pd
import time

def CreateInterventions(layer_id,layer,pos_type,rep_type,outputs,token_ids):
    assert pos_type in ['choice','context'] and rep_type in ['layer','key','query','value']
    if pos_type=='choice':
        if rep_type=='layer':
            vec_correct = outputs[1][layer_id][0,token_ids[f'choice_correct']]
            vec_incorrect = outputs[1][layer_id][0,token_ids[f'choice_incorrect']]
        elif rep_type=='key':
            key = layer.attention.self.key(outputs[1][layer_id])
            vec_correct = key[0,token_ids[f'choice_correct']]
            vec_incorrect = key[0,token_ids[f'choice_incorrect']]
        elif rep_type=='query':
            query = layer.attention.self.query(outputs[1][layer_id])
            vec_correct = query[0,token_ids[f'choice_correct']]
            vec_incorrect = query[0,token_ids[f'choice_incorrect']]
        elif rep_type=='value':
            value = layer.attention.self.value(outputs[1][layer_id])
            vec_correct = value[0,token_ids[f'choice_correct']]
            vec_incorrect = value[0,token_ids[f'choice_incorrect']]
        else:
            raise NotImplementedError(f'rep_type "{rep_type}" is not supported')
        return vec_correct, vec_incorrect
    elif pos_type=='context':
        if rep_type=='layer':
            vec = outputs[1][layer_id][0,token_ids[f'context']]
        elif rep_type=='key':
            key = layer.attention.self.key(outputs[1][layer_id])
            vec = key[0,token_ids[f'context']]
        elif rep_type=='query':
            query = layer.attention.self.query(outputs[1][layer_id])
            vec = query[0,token_ids[f'context']]
        elif rep_type=='value':
            value = layer.attention.self.value(outputs[1][layer_id])
            vec = value[0,token_ids[f'context']]
        else:
            raise NotImplementedError(f'rep_type "{rep_type}" is not supported')
        return vec

def ApplyInterventionsLayer(model,layer_id,layer,pos_types,rep_types,outputs,token_ids,choice_tokens_lists,args,verbose=False):
    # Unpack the input
    token_ids_1 = token_ids[0]
    token_ids_2 = token_ids[1]
    choice_tokens_list_1 = choice_tokens_lists[0]
    choice_tokens_list_2 = choice_tokens_lists[1]

    interventions_1 = {'correct':{},'incorrect':{}}
    for sent_1_type,sent_2_type in zip(['correct', 'incorrect'],['incorrect','correct']):
        for pos_type in pos_types:
            if pos_type=='choice':
                pos_correct_1 = token_ids_1[f'{sent_1_type}_sent']['choice_correct']
                pos_incorrect_1 = token_ids_1[f'{sent_1_type}_sent']['choice_incorrect']
                for rep_type in rep_types:
                    vec_correct_2,vec_incorrect_2 = CreateInterventions(layer_id,layer,pos_type,rep_type,outputs[f'{sent_2_type}_2'],token_ids_2[f'{sent_2_type}_sent'])
                    assert len(pos_correct_1)==len(vec_incorrect_2) and len(pos_incorrect_1)==len(vec_correct_2)
                    if f'{rep_type}_{layer_id}' not in interventions_1[sent_1_type]:
                        interventions_1[sent_1_type][f'{rep_type}_{layer_id}'] = []
                    interventions_1[sent_1_type][f'{rep_type}_{layer_id}'].extend([(pos_correct_1,vec_incorrect_2),(pos_incorrect_1,vec_correct_2)])
            elif pos_type=='context':
                pos_context_1 = token_ids_1[f'{sent_1_type}_sent']['context']
                for rep_type in rep_types:
                    vec_2 = CreateInterventions(layer_id,layer,pos_type,rep_type,outputs[f'{sent_2_type}_2'],token_ids_2[f'{sent_2_type}_sent'])
                    if f'{rep_type}_{layer_id}' not in interventions_1[sent_1_type]:
                        interventions_1[sent_1_type][f'{rep_type}_{layer_id}'] = []
                    interventions_1[sent_1_type][f'{rep_type}_{layer_id}'].extend([(pos_context_1,vec_2)])

    interventions_2 = {'correct':{},'incorrect':{}}
    for sent_2_type,sent_1_type in zip(['correct', 'incorrect'],['incorrect','correct']):
        for pos_type in pos_types:
            if pos_type=='choice':
                pos_correct_2 = token_ids_2[f'{sent_2_type}_sent']['choice_correct']
                pos_incorrect_2 = token_ids_2[f'{sent_2_type}_sent']['choice_incorrect']
                for rep_type in rep_types:
                    vec_correct_1,vec_incorrect_1 = CreateInterventions(layer_id,layer,pos_type,rep_type,outputs[f'{sent_1_type}_1'],token_ids_1[f'{sent_1_type}_sent'])
                    assert len(pos_correct_2)==len(vec_incorrect_1) and len(pos_incorrect_2)==len(vec_correct_1)
                    if f'{rep_type}_{layer_id}' not in interventions_2[sent_2_type]:
                        interventions_2[sent_2_type][f'{rep_type}_{layer_id}'] = []
                    interventions_2[sent_2_type][f'{rep_type}_{layer_id}'].extend([(pos_correct_2,vec_incorrect_1),(pos_incorrect_2,vec_correct_1)])
            elif pos_type=='context':
                pos_context_2 = token_ids_2[f'{sent_2_type}_sent']['context']
                for rep_type in rep_types:
                    vec_1 = CreateInterventions(layer_id,layer,pos_type,rep_type,outputs[f'{sent_1_type}_1'],token_ids_1[f'{sent_1_type}_sent'])
                    if f'{rep_type}_{layer_id}' not in interventions_2[sent_2_type]:
                        interventions_2[sent_2_type][f'{rep_type}_{layer_id}'] = []
                    interventions_2[sent_2_type][f'{rep_type}_{layer_id}'].extend([(pos_context_2,vec_1)])

    if verbose:
        for interventions in [interventions_1['correct'],interventions_1['incorrect'],interventions_2['correct'],interventions_2['incorrect']]:
            for key,value in interventions.items():
                print(key)
                for pair in value:
                    print(pair[0],pair[1].shape)

    int_logits_correct_1 = skeleton_model(layer_id,outputs['correct_1'][1][layer_id],model,args.model,interventions_1['correct'],args)
    int_logits_incorrect_1 = skeleton_model(layer_id,outputs['incorrect_1'][1][layer_id],model,args.model,interventions_1['incorrect'],args)
    int_logits_correct_2 = skeleton_model(layer_id,outputs['correct_2'][1][layer_id],model,args.model,interventions_2['correct'],args)
    int_logits_incorrect_2 = skeleton_model(layer_id,outputs['incorrect_2'][1][layer_id],model,args.model,interventions_2['incorrect'],args)

    choice_probs_sum_1,choice_probs_ave_1 = EvaluatePredictions(int_logits_correct_1,int_logits_incorrect_1,token_ids_1,choice_tokens_list_1,args)
    choice_probs_sum_2,choice_probs_ave_2 = EvaluatePredictions(int_logits_correct_2,int_logits_incorrect_2,token_ids_2,choice_tokens_list_2,args)

    return choice_probs_sum_1,choice_probs_ave_1,choice_probs_sum_2,choice_probs_ave_2


def ApplyInterventions(head,line,pos_types,rep_types,model,tokenizer,mask_id,args):
    # the first sentence should be the one with the correct NP being prior to the incorrect NP
    if int(line[head.index('choice_word_id_correct_1')])<int(line[head.index('choice_word_id_incorrect_1')]):
        assert int(line[head.index('choice_word_id_correct_2')])>int(line[head.index('choice_word_id_incorrect_2')])
        outputs_correct_1,outputs_incorrect_1,token_ids_1,choice_tokens_list_1,_ = CalcOutputs(head,line,1,model,tokenizer,mask_id,args)
        outputs_correct_2,outputs_incorrect_2,token_ids_2,choice_tokens_list_2,_ = CalcOutputs(head,line,2,model,tokenizer,mask_id,args)

    elif int(line[head.index('choice_word_id_correct_1')])>int(line[head.index('choice_word_id_incorrect_1')]):
        assert int(line[head.index('choice_word_id_correct_2')])<int(line[head.index('choice_word_id_incorrect_2')])
        outputs_correct_1,outputs_incorrect_1,token_ids_1,choice_tokens_list_1,_ = CalcOutputs(head,line,2,model,tokenizer,mask_id,args)
        outputs_correct_2,outputs_incorrect_2,token_ids_2,choice_tokens_list_2,_ = CalcOutputs(head,line,1,model,tokenizer,mask_id,args)

    outputs = {}
    outputs['correct_1'] = outputs_correct_1
    outputs['incorrect_1'] = outputs_incorrect_1
    outputs['correct_2'] = outputs_correct_2
    outputs['incorrect_2'] = outputs_incorrect_2

    token_ids = [token_ids_1, token_ids_2]
    choice_tokens_lists = [choice_tokens_list_1,choice_tokens_list_2]

    if args.model.startswith('bert'):
        core_model = model.bert
        lm_head = model.cls
    elif args.model.startswith('roberta'):
        core_model = model.roberta
        lm_head = model.lm_head

    out_dict = {}
    choice_probs_sum_1, choice_probs_ave_1 = EvaluatePredictions(outputs_correct_1[0],outputs_incorrect_1[0],token_ids_1,choice_tokens_list_1,args)
    choice_probs_sum_2, choice_probs_ave_2 = EvaluatePredictions(outputs_correct_2[0],outputs_incorrect_2[0],token_ids_2,choice_tokens_list_2,args)
    out_dict['original'] = {}
    out_dict['original']['sum_1'] = choice_probs_sum_1
    out_dict['original']['sum_2'] = choice_probs_sum_2
    out_dict['original']['ave_1'] = choice_probs_ave_1
    out_dict['original']['ave_2'] = choice_probs_ave_2

    for layer_id, layer in enumerate(core_model.encoder.layer):
        int_choice_probs_sum_1,int_choice_probs_ave_1,int_choice_probs_sum_2,int_choice_probs_ave_2 = ApplyInterventionsLayer(model,layer_id,layer,pos_types,rep_types,outputs,token_ids,choice_tokens_lists,args)
        out_dict[f'layer_{layer_id}'] = {}
        out_dict[f'layer_{layer_id}']['sum_1'] = int_choice_probs_sum_1
        out_dict[f'layer_{layer_id}']['sum_2'] = int_choice_probs_sum_2
        out_dict[f'layer_{layer_id}']['ave_1'] = int_choice_probs_ave_1
        out_dict[f'layer_{layer_id}']['ave_2'] = int_choice_probs_ave_2

    return out_dict


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
    parser.add_argument('--pos_type', type = str, required = True)
    args = parser.parse_args()
    # context and choice perturbations should be in this order, not the other way round
    assert args.pos_type in ['choice','context','choice_context']
    print(f'running with {args}')

    head,text = LoadDataset(args)
    model, tokenizer, mask_id, args = LoadModel(args)

    out_dict = {}
    for line in text[:500]:
        interventions = {}
        results = ApplyInterventions(head,line,args.pos_type.split('_'),args.rep_type.split('_'),model,tokenizer,mask_id,args)
        out_dict[line[head.index('pair_id')]] = results

    if args.dataset=='superglue':
        with open(f'datafile/superglue_wsc_intervention_{args.pos_type}_{args.rep_type}_{args.model}_{args.stimuli}.pkl','wb') as f:
            pickle.dump(out_dict,f)
    elif args.dataset=='winogrande':
        with open(f'datafile/winogrande_{args.size}_intervention_{args.pos_type}_{args.rep_type}_{args.model}.pkl','wb') as f:
            pickle.dump(out_dict,f)

    print(f'Time it took: {time.time()-start}')
