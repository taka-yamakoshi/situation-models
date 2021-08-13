import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import argparse
import json
import csv
from SuperGLUE_WSC_alignment import AlignPronoun,AlignCandidates,AlignContext,AlignPeriod
from SuperGLUE_WSC_attention import convert_to_numpy
import time
import networkx as nx
from multiprocessing import Pool

def CreateGraph(input_attn):
    num_layers = input_attn.shape[0]
    seq_len = input_attn.shape[-1]
    adj_mat = np.zeros(((num_layers+1)*seq_len, (num_layers+1)*seq_len))
    for layer_id in range(1,num_layers+1):
        for pos_from in range(seq_len):
            for pos_to in range(seq_len):
                adj_mat[layer_id*seq_len+pos_from][(layer_id-1)*seq_len+pos_to] = input_attn[layer_id-1][pos_from][pos_to]
    G=nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph())
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            nx.set_edge_attributes(G, {(i,j): adj_mat[i,j]}, 'capacity')
    return G

def CalcAttnFlowNode(G,input_attn,pos_from):
    attn_flow = np.empty((input_attn.shape[0],input_attn.shape[-1]))
    for layer_id in range(input_attn.shape[0]):
        u = (layer_id+1)*input_attn.shape[-1]+pos_from
        for pos_to in range(input_attn.shape[-1]):
            attn_flow[layer_id][pos_to] = MyMaxFlow(G,u,pos_to)
        #arg = [(G,u,pos_to) for pos_to in range(input_attn.shape[-1])]
        #with Pool(processes=10) as p:
        #    result_list = p.starmap(MyMaxFlow,arg)
        #attn_flow[layer_id] = result_list
    return attn_flow/attn_flow.sum(axis=1)[...,None]

def MyMaxFlow(G,u,pos_to):
    return nx.maximum_flow_value(G,u,pos_to,flow_func=nx.algorithms.flow.edmonds_karp)

def CalcAttnFlow(head,line,sent_id):
    # load data from a line in csv
    sent = line[head.index(f'sent_{sent_id}')]
    pron = line[head.index(f'pron_{sent_id}')]
    pron_word_id = int(line[head.index(f'pron_word_id_{sent_id}')])
    choice_correct = line[head.index(f'choice_correct_{sent_id}')]
    choice_incorrect = line[head.index(f'choice_incorrect_{sent_id}')]
    choice_word_id_correct = int(line[head.index(f'choice_word_id_correct_{sent_id}')])
    choice_word_id_incorrect = int(line[head.index(f'choice_word_id_incorrect_{sent_id}')])
    context = line[head.index(f'context_{sent_id}')]
    context_word_id = int(line[head.index(f'context_word_id_{sent_id}')])

    input_sent = tokenizer(sent,return_tensors='pt')['input_ids']
    pron_token_id = AlignPronoun(tokenizer,sent,input_sent,pron,pron_word_id)
    choice_start_id_correct,choice_end_id_correct = AlignCandidates(tokenizer,sent,input_sent,choice_correct,choice_word_id_correct)
    choice_start_id_incorrect,choice_end_id_incorrect = AlignCandidates(tokenizer,sent,input_sent,choice_incorrect,choice_word_id_incorrect)
    context_start_id,context_end_id = AlignContext(tokenizer,sent,input_sent,context,context_word_id)
    period_id = AlignPeriod(tokenizer,sent,input_sent)

    choice_tokens_list = [input_sent[0][choice_start_id_correct:choice_end_id_correct],
    input_sent[0][choice_start_id_incorrect:choice_end_id_incorrect]]

    masked_sent = input_sent.clone()
    masked_sent[0][pron_token_id] = mask_id
    with torch.no_grad():
        outputs = model(input_sent.to(args.device))
        outputs_masked = model(masked_sent.to(args.device))
        attention = convert_to_numpy(outputs[-1])
        attention_masked = convert_to_numpy(outputs_masked[-1])

    if args.residual:
        attn = 0.5*attention+0.5*np.eye(attention.shape[-1])[None,None,...]
        attn_masked = 0.5*attention_masked+0.5*np.eye(attention_masked.shape[-1])[None,None,...]
    else:
        attn = attention
        attn_masked = attention_masked

    attn_graph = CreateGraph(attn.mean(axis=1))
    attn_masked_graph = CreateGraph(attn_masked.mean(axis=1))
    start = time.time()
    #arg = [(attn_graph,attn.mean(axis=1),pron_token_id),(attn_masked_graph,attn_masked.mean(axis=1),pron_token_id)]
    #with Pool() as p:
    #    [attn_flow,attn_masked_flow] = p.starmap(CalcAttnFlowNode,arg)
    attn_flow = CalcAttnFlowNode(attn_graph,attn.mean(axis=1),pron_token_id)
    attn_masked_flow = CalcAttnFlowNode(attn_masked_graph,attn_masked.mean(axis=1),pron_token_id)
    print(time.time() - start)

    attn_choice_correct = attn_flow[:,choice_start_id_correct:choice_end_id_correct].sum(axis=-1)
    attn_choice_incorrect = attn_flow[:,choice_start_id_incorrect:choice_end_id_incorrect].sum(axis=-1)
    attn_masked_choice_correct = attn_masked_flow[:,choice_start_id_correct:choice_end_id_correct].sum(axis=-1)
    attn_masked_choice_incorrect = attn_masked_flow[:,choice_start_id_incorrect:choice_end_id_incorrect].sum(axis=-1)
    attn_context_pron = attn_flow[:,context_start_id:context_end_id].sum(axis=-1)
    attn_masked_context_pron = attn_masked_flow[:,context_start_id:context_end_id].sum(axis=-1)
    attn_period = attn_flow[:,period_id]
    attn_masked_period = attn_masked_flow[:,period_id]

    attn_context_dict = {}
    attn_context_dict['pron'] = attn_context_pron
    attn_masked_context_dict = {}
    attn_masked_context_dict['pron'] = attn_masked_context_pron
    return np.array([attn_choice_correct,attn_choice_incorrect]),np.array([attn_masked_choice_correct,attn_masked_choice_incorrect]),attn_context_dict,attn_masked_context_dict,attn_period,attn_masked_period

def RunParallel(head,line):
    attn_choice_1,attn_choice_masked_1,attn_context_dict_1,attn_masked_context_dict_1,attn_period_1,attn_masked_period_1 = CalcAttnFlow(head,line,1)
    attn_choice_2,attn_choice_masked_2,attn_context_dict_2,attn_masked_context_dict_2,attn_period_2,attn_masked_period_2 = CalcAttnFlow(head,line,2)
    return ((attn_choice_1,attn_choice_masked_1,attn_context_dict_1,attn_masked_context_dict_1,attn_period_1,attn_masked_period_1),
            (attn_choice_2,attn_choice_masked_2,attn_context_dict_2,attn_masked_context_dict_2,attn_period_2,attn_masked_period_2))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--stimuli', type = str,
                        choices=['original','control_gender','control_number','control_combined'],
                        default='original')
    parser.add_argument('--core_id', type = int, default=0)
    parser.add_argument('--residual',dest='residual',action='store_true')
    parser.set_defaults(residual=False)
    args = parser.parse_args()
    print(f'running with {args}')

    if args.residual:
        res_id = '_res'
    else:
        res_id = ''

    # load the csv file
    if args.stimuli=='original':
        fname = 'datafile/SuperGLUE_wsc_new.csv'
    elif args.stimuli=='control_gender':
        fname = 'datafile/SuperGLUE_wsc_new_control_gender.csv'
    elif args.stimuli=='control_number':
        fname = 'datafile/SuperGLUE_wsc_new_control_number.csv'
    elif args.stimuli=='control_combined':
        fname = 'datafile/SuperGLUE_wsc_new_control_combined.csv'

    with open(fname,'r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
    head = file[0]
    text = file[1:]

    # load the model
    if 'bert-' in args.model:
        from transformers import BertTokenizer, BertForMaskedLM, BertModel
        model = BertForMaskedLM.from_pretrained(args.model,output_hidden_states=True,output_attentions=True)
        tokenizer = BertTokenizer.from_pretrained(args.model)
    elif 'roberta-' in args.model:
        from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
        model = RobertaForMaskedLM.from_pretrained(args.model,output_hidden_states=True,output_attentions=True)
        tokenizer = RobertaTokenizer.from_pretrained(args.model)
    if torch.cuda.is_available():
        args.device = torch.device("cuda", index=int(args.core_id))
    else:
        args.device = torch.device("cpu")
    model.to(args.device)
    model.eval()
    mask_id = tokenizer.encode("[MASK]")[1:-1][0]

    arg = [(head,line) for line in text]
    args.device='cpu'
    model.to(args.device)
    model.eval()
    with Pool() as p:
        result_list = p.starmap(RunParallel,arg)
    assert len(result_list)==len(text)

    out_dict = {}
    for row,line in zip(result_list,text):
        (attn_choice_1,attn_choice_masked_1,attn_context_dict_1,attn_masked_context_dict_1,attn_period_1,attn_masked_period_1) = row[0]
        (attn_choice_2,attn_choice_masked_2,attn_context_dict_2,attn_masked_context_dict_2,attn_period_2,attn_masked_period_2) = row[1]
        out_dict[line[head.index('pair_id')]] = {}
        out_dict[line[head.index('pair_id')]]['choice_1'] = attn_choice_1
        out_dict[line[head.index('pair_id')]]['choice_2'] = attn_choice_2
        out_dict[line[head.index('pair_id')]]['choice_masked_1'] = attn_choice_masked_1
        out_dict[line[head.index('pair_id')]]['choice_masked_2'] = attn_choice_masked_2
        out_dict[line[head.index('pair_id')]]['context_1'] = attn_context_dict_1
        out_dict[line[head.index('pair_id')]]['context_2'] = attn_context_dict_2
        out_dict[line[head.index('pair_id')]]['context_masked_1'] = attn_masked_context_dict_1
        out_dict[line[head.index('pair_id')]]['context_masked_2'] = attn_masked_context_dict_2
        out_dict[line[head.index('pair_id')]]['period_1'] = attn_period_1
        out_dict[line[head.index('pair_id')]]['period_2'] = attn_period_2
        out_dict[line[head.index('pair_id')]]['period_masked_1'] = attn_masked_period_1
        out_dict[line[head.index('pair_id')]]['period_masked_2'] = attn_masked_period_2



    #for line in text:
        #if 'willow-towered Canopy Huntertropic wrestles' in [line[head.index('choice_correct_1')],line[head.index('choice_correct_2')]]:
        #    print('passed 1')
        #    continue
        #elif 'My great-grandfather' in [line[head.index('choice_correct_1')],line[head.index('choice_correct_2')]]:
        #    print('passed 2')
        #    continue

        #attn_choice_1,attn_choice_masked_1,attn_context_dict_1,attn_masked_context_dict_1,attn_period_1,attn_masked_period_1 = CalcAttnFlow(head,line,1)
        #attn_choice_2,attn_choice_masked_2,attn_context_dict_2,attn_masked_context_dict_2,attn_period_2,attn_masked_period_2 = CalcAttnFlow(head,line,2)

        #out_dict[line[head.index('pair_id')]] = {}
        #out_dict[line[head.index('pair_id')]]['choice_1'] = attn_choice_1
        #out_dict[line[head.index('pair_id')]]['choice_2'] = attn_choice_2
        #out_dict[line[head.index('pair_id')]]['choice_masked_1'] = attn_choice_masked_1
        #out_dict[line[head.index('pair_id')]]['choice_masked_2'] = attn_choice_masked_2
        #out_dict[line[head.index('pair_id')]]['context_1'] = attn_context_dict_1
        #out_dict[line[head.index('pair_id')]]['context_2'] = attn_context_dict_2
        #out_dict[line[head.index('pair_id')]]['context_masked_1'] = attn_masked_context_dict_1
        #out_dict[line[head.index('pair_id')]]['context_masked_2'] = attn_masked_context_dict_2
        #out_dict[line[head.index('pair_id')]]['period_1'] = attn_period_1
        #out_dict[line[head.index('pair_id')]]['period_2'] = attn_period_2
        #out_dict[line[head.index('pair_id')]]['period_masked_1'] = attn_masked_period_1
        #out_dict[line[head.index('pair_id')]]['period_masked_2'] = attn_masked_period_2

    with open(f'datafile/superglue_wsc_attention_flow_{args.model}_{args.stimuli}{res_id}.pkl','wb') as f:
        pickle.dump(out_dict,f)
