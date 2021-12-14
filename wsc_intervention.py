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

def ExtractLayer(layer_id,model,args):
    if args.model.startswith('bert'):
        layer = model.bert.encoder.layer[layer_id]
    elif args.model.startswith('roberta'):
        layer = model.roberta.encoder.layer[layer_id]
    else:
        raise NotImplementedError("invalid model name")
    return layer

def CreateInterventions(layer_id,pos_type,rep_type,outputs,token_ids,args):
    assert pos_type in ['choice','context','masks','period','cls','sep','other'] and rep_type in ['layer','key','query','value']
    if pos_type=='choice':
        if rep_type=='layer':
            vec_option_1 = outputs[1][layer_id][0,token_ids[f'option_1']]
            vec_option_2 = outputs[1][layer_id][0,token_ids[f'option_2']]
        elif rep_type in ['key','query','value']:
            layer = ExtractLayer(layer_id,model,args)
            if rep_type=='key':
                key = layer.attention.self.key(outputs[1][layer_id])
                vec_option_1 = key[0,token_ids[f'option_1']]
                vec_option_2 = key[0,token_ids[f'option_2']]
            elif rep_type=='query':
                query = layer.attention.self.query(outputs[1][layer_id])
                vec_option_1 = query[0,token_ids[f'option_1']]
                vec_option_2 = query[0,token_ids[f'option_2']]
            elif rep_type=='value':
                value = layer.attention.self.value(outputs[1][layer_id])
                vec_option_1 = value[0,token_ids[f'option_1']]
                vec_option_2 = value[0,token_ids[f'option_2']]
        else:
            raise NotImplementedError(f'rep_type "{rep_type}" is not supported')
        return vec_option_1, vec_option_2
    elif pos_type in ['context','masks','period','cls','sep','other']:
        if rep_type=='layer':
            vec = outputs[1][layer_id][0,token_ids[f'{pos_type}']]
        elif rep_type in ['key','query','value']:
            layer = ExtractLayer(layer_id,model,args)
            if rep_type=='key':
                key = layer.attention.self.key(outputs[1][layer_id])
                vec = key[0,token_ids[f'{pos_type}']]
            elif rep_type=='query':
                query = layer.attention.self.query(outputs[1][layer_id])
                vec = query[0,token_ids[f'{pos_type}']]
            elif rep_type=='value':
                value = layer.attention.self.value(outputs[1][layer_id])
                vec = value[0,token_ids[f'{pos_type}']]
        else:
            raise NotImplementedError(f'rep_type "{rep_type}" is not supported')
        return vec

def ApplyInterventionsLayer(model,layer_id,pos_types,rep_types,outputs,token_ids,option_tokens_lists,args,verbose=False):
    interventions_all = []
    for context_id in range(2):
        interventions = {'masked_sent_1':{},'masked_sent_2':{}}
        for masked_sent_id in [1,2]:
            for pos_type in pos_types:
                if pos_type=='choice':
                    pos_option_1 = token_ids[context_id][f'masked_sent_{masked_sent_id}']['option_1']
                    pos_option_2 = token_ids[context_id][f'masked_sent_{masked_sent_id}']['option_2']
                    for rep_type in rep_types:
                        if args.test:
                            vec_option_1,vec_option_2 = CreateInterventions(layer_id,pos_type,rep_type,
                                                                            outputs[f'masked_sent_{masked_sent_id}_context_{context_id+1}'],
                                                                            token_ids[context_id][f'masked_sent_{masked_sent_id}'],args)
                        else:
                            vec_option_1,vec_option_2 = CreateInterventions(layer_id,pos_type,rep_type,
                                                                            outputs[f'masked_sent_{masked_sent_id}_context_{2-context_id}'],
                                                                            token_ids[1-context_id][f'masked_sent_{masked_sent_id}'],args)
                        assert len(pos_option_1)==len(vec_option_1)
                        assert len(pos_option_2)==len(vec_option_2)
                        if f'{rep_type}_{layer_id}' not in interventions[f'masked_sent_{masked_sent_id}']:
                            interventions[f'masked_sent_{masked_sent_id}'][f'{rep_type}_{layer_id}'] = []
                        interventions[f'masked_sent_{masked_sent_id}'][f'{rep_type}_{layer_id}'].extend([(pos_option_1,vec_option_1),(pos_option_2,vec_option_2)])
                elif pos_type in ['context','masks','period','cls','sep','other']:
                    pos = token_ids[context_id][f'masked_sent_{masked_sent_id}'][f'{pos_type}']
                    for rep_type in rep_types:
                        if args.test:
                            vec = CreateInterventions(layer_id,pos_type,rep_type,
                                                    outputs[f'masked_sent_{masked_sent_id}_context_{context_id+1}'],
                                                    token_ids[context_id][f'masked_sent_{masked_sent_id}'],args)
                        else:
                            vec = CreateInterventions(layer_id,pos_type,rep_type,
                                                    outputs[f'masked_sent_{masked_sent_id}_context_{2-context_id}'],
                                                    token_ids[1-context_id][f'masked_sent_{masked_sent_id}'],args)
                        if pos_type!='context':
                            assert len(pos)==len(vec)
                        if f'{rep_type}_{layer_id}' not in interventions[f'masked_sent_{masked_sent_id}']:
                            interventions[f'masked_sent_{masked_sent_id}'][f'{rep_type}_{layer_id}'] = []
                        interventions[f'masked_sent_{masked_sent_id}'][f'{rep_type}_{layer_id}'].extend([(pos,vec)])
                else:
                    raise NotImplementedError(f'pos_type "{pos_type}" is not supported')
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

    if 'context' in pos_types:
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
        int_choice_probs_sum_1,int_choice_probs_ave_1,int_choice_probs_sum_2,int_choice_probs_ave_2 = ApplyInterventionsLayer(model,layer_id,pos_types,rep_types,outputs,token_ids,option_tokens_lists,args)
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
    parser.add_argument('--test',dest='test',action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()
    # context and masks perturbations should be the last since they may change the sentence length
    assert args.rep_type in ['layer','layer_query','key_value','layer_query_key_value']
    assert args.pos_type in ['choice','context','masks','period','cls','sep','cls_sep','cls_period_sep','choice_context','other']
    print(f'running with {args}')

    if args.test:
        test_id = '_test'
    else:
        test_id = ''

    head,text = LoadDataset(args)
    model, tokenizer, mask_id, args = LoadModel(args)

    out_dict = {}
    for line in text[:500]:
        results = ApplyInterventions(head,line,args.pos_type.split('_'),args.rep_type.split('_'),model,tokenizer,mask_id,args)
        out_dict[line[head.index('pair_id')]] = results

    if args.dataset=='superglue':
        with open(f'datafile/superglue_wsc_intervention_{args.pos_type}_{args.rep_type}_{args.model}_{args.stimuli}{test_id}.pkl','wb') as f:
            pickle.dump(out_dict,f)
    elif args.dataset=='winogrande':
        with open(f'datafile/winogrande_{args.size}_intervention_{args.pos_type}_{args.rep_type}_{args.model}{test_id}.pkl','wb') as f:
            pickle.dump(out_dict,f)

    print(f'Time it took: {time.time()-start}')
