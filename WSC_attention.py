import json
import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertModel
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import itertools
import networkx as nx
import pickle
import os
import argparse

def calc_attention_rollout(attn):
    full_attn = [0.5*layer.squeeze()+0.5*torch.eye(layer.squeeze().shape[-1])[None,...] for layer in attn]
    ave_attn = [layer.mean(dim=0) for layer in full_attn]
    attn_rollout = [full_attn[0]]
    ave_attn_rollout = [ave_attn[0]]
    for layer,ave_layer in zip(full_attn[1:],ave_attn[1:]):
        layer_rollout = torch.tensor([list((head@ave_attn_rollout[-1]).detach().numpy()) for head in layer])
        attn_rollout.append(layer_rollout)
        ave_attn_rollout.append(ave_layer@ave_attn_rollout[-1])
    return convert_to_numpy(attn_rollout)

def calc_attention_flow(attn):
    full_attn = [0.5*layer.squeeze()+0.5*torch.eye(layer.squeeze().shape[-1])[None,...] for layer in attn]
    ave_attn = [layer.mean(dim=0) for layer in full_attn]
    num_layers = len(ave_attn)
    seq_len = ave_attn[0].shape[0]
    # create graph
    adj_mat = np.zeros(((num_layers+1)*seq_len, (num_layers+1)*seq_len))
    for layer_id in range(1,num_layers+1):
        for pos_from in range(seq_len):
            for pos_to in range(seq_len):
                adj_mat[layer_id*seq_len+pos_from][(layer_id-1)*seq_len+pos_to] = ave_attn[layer_id-1][pos_from][pos_to]
    G=nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph())
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            nx.set_edge_attributes(G, {(i,j): adj_mat[i,j]}, 'capacity')
    # calculate max flow
    max_flows = []
    for layer_id in range(1,num_layers+1):
        max_flow_layer = np.zeros((seq_len,seq_len))
        for pos in range(seq_len):
            for input_node in range(seq_len):
                max_flow_layer[pos,input_node] = nx.maximum_flow_value(G,layer_id*seq_len+pos,input_node, flow_func=nx.algorithms.flow.edmonds_karp)
        max_flows.append(torch.from_numpy(max_flow_layer).float())
    normed_max_flows = [layer/layer.sum(dim=1)[...,None] for layer in max_flows]
    for layer in normed_max_flows:
        assert torch.allclose(layer.sum(dim=1),torch.ones_like(layer.sum(dim=1)))
    attn_flow = [full_attn[0]]
    for layer, ave_layer in zip(full_attn[1:],normed_max_flows[:-1]):
        layer_flow = torch.tensor([list((head@ave_layer).detach().numpy()) for head in layer])
        attn_flow.append(layer_flow)
    return convert_to_numpy(attn_flow)

def convert_to_numpy(attn):
    return np.array([layer.squeeze().detach().numpy() for layer in attn])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    args = parser.parse_args()

    model = BertForMaskedLM.from_pretrained(args.model,output_hidden_states=True,output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model.eval()

    with open('datafile/wsc_data_new.pkl','rb') as f:
        wsc_data = pickle.load(f)

    print(f'# sentence pairs used: {len(list(wsc_data.keys()))}')


    schema_id = 0
    out_dict = {}
    os.makedirs(f'figures/attention_diff/{args.model}/',exist_ok=True)
    for key,value in wsc_data.items():
        assert len(value)==2
        print(value)
        sent_1 = value[0]['sent']
        sent_2 = value[1]['sent']
        print('sentences:')
        print(sent_1)
        print(sent_2)
        input_1 = tokenizer(sent_1,return_tensors='pt')['input_ids']
        input_2 = tokenizer(sent_2,return_tensors='pt')['input_ids']
        print('input_tensors:')
        print(input_1)
        print(input_2)
        attention_1 = model(input_1)[2]
        attention_2 = model(input_2)[2]
        attn_mat_1 = convert_to_numpy(attention_1)
        attn_mat_2 = convert_to_numpy(attention_2)
        #attn_rollout_1 = calc_attention_rollout(attention_1)
        #attn_rollout_2 = calc_attention_rollout(attention_2)
        #print("calculating attention flow 1")
        #attn_flow_1 = calc_attention_flow(attention_1)
        #print("calculating attention flow 2")
        #attn_flow_2 = calc_attention_flow(attention_2)
        #print('attention shapes: ')
        #print(attn_mat_2.shape,attn_rollout_2.shape,attn_flow_2.shape)

        pron_1 = value[0]['pron']
        pron_2 = value[1]['pron']
        assert pron_1==pron_2
        pron_token = tokenizer(pron_1)['input_ids'][1:-1]
        print('pronoun:')
        print(pron_1,pron_2)
        print(pron_token)
        assert len(pron_token)==1
        assert np.sum([token==pron_token[0] for token in input_1[0]])==1
        assert np.sum([token==pron_token[0] for token in input_2[0]])==1

        # Extract weights related to predicting the pronoun
        attn_row_1 = attn_mat_1[:,:,list(input_1[0]).index(pron_token[0]),:]
        attn_row_2 = attn_mat_2[:,:,list(input_2[0]).index(pron_token[0]),:]

        choices_1 = value[0]['choices']
        choices_2 = value[1]['choices']
        assert choices_1[0]==choices_2[0] and choices_1[1]==choices_2[1]
        choices = choices_1
        print('choices')
        print(choices)

        choice_tokens_list = [tokenizer(choice)['input_ids'][1:-1] for choice in choices]
        print(choice_tokens_list)

        # See where choice tokesn appear in each sentence
        choice_sent_1_list = [[[token_id for token_id,token in enumerate(input_1[0]) if token==choice_token]
                               for choice_token in choice_tokens]
                              for choice_tokens in choice_tokens_list]
        choice_sent_2_list = [[[token_id for token_id,token in enumerate(input_2[0]) if token==choice_token]
                               for choice_token in choice_tokens]
                              for choice_tokens in choice_tokens_list]
        print(choice_sent_1_list)
        print(choice_sent_2_list)

        # Consider all possible conbinations
        choice_sent_1_all = [list(itertools.product(*line)) for line in choice_sent_1_list]
        choice_sent_2_all = [list(itertools.product(*line)) for line in choice_sent_2_list]
        print(choice_sent_1_all)
        print(choice_sent_2_all)

        # Choose a combination that is sequential
        choice_sent_1_ids = [[list(row) for row in line if np.all(np.diff(list(row))==1)] for line in choice_sent_1_all]
        choice_sent_2_ids = [[list(row) for row in line if np.all(np.diff(list(row))==1)] for line in choice_sent_2_all]
        assert len(choice_sent_1_ids[0])==1 and len(choice_sent_2_ids[0])==1
        assert len(choice_sent_1_ids[1])==1 and len(choice_sent_2_ids[1])==1
        choice_sent_1_ids = [line[0] for line in choice_sent_1_ids]
        choice_sent_2_ids = [line[0] for line in choice_sent_2_ids]
        print(choice_sent_1_ids)
        print(choice_sent_2_ids)

        ref_weight_1 = np.array([attn_row_1[:,:,choice_ids].sum(axis=-1) for choice_ids in choice_sent_1_ids])
        ref_weight_2 = np.array([attn_row_2[:,:,choice_ids].sum(axis=-1) for choice_ids in choice_sent_2_ids])

        correct_1 = value[0]['correct_ans']
        correct_2 = value[1]['correct_ans']
        print(correct_1)
        print(correct_2)
        correct_id_1 = ['A','B'].index(correct_1)
        correct_id_2 = ['A','B'].index(correct_2)
        print(correct_id_1,correct_id_2)

        fig = plt.figure(figsize=(20,2))
        ax = fig.add_subplot(1,1,1)
        ax.scatter([i+i//12 for i in range(144)],[head for layer in ref_weight_1[correct_id_1]-ref_weight_1[correct_id_2] for head in layer],alpha=0.5,marker='.',label=f'sent_1')
        ax.scatter([i+i//12 for i in range(144)],[head for layer in ref_weight_2[correct_id_1]-ref_weight_2[correct_id_2] for head in layer],alpha=0.5,marker='.',label=f'sent_2')
        ax.axhline(y=0,xmin=0,xmax=144+12)
        ax.set_xticks([i+i//12 for i in range(144)])
        ax.set_xticklabels([i if i%2==0 else None for _ in range(12) for i in range(12)])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.legend(frameon=False,bbox_to_anchor=(1.02, 1.1))
        ax.set_title(f'{sent_1} / {sent_2}')
        fig.savefig(f'figures/attention_diff/{args.model}/attention_diff_{schema_id}.png')
        schema_id += 1

        out_dict[key] = np.array([[ref_weight_1[correct_id_1],ref_weight_1[correct_id_2]],[ref_weight_2[correct_id_2],ref_weight_2[correct_id_1]]])

    with open(f'datafile/wsc_attention_{args.model}.pkl','wb') as f:
        pickle.dump(out_dict,f)
