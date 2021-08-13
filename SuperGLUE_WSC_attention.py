import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import argparse
import json
import csv
from SuperGLUE_WSC_alignment import AlignPronoun,AlignCandidates,AlignContext,AlignPeriod

def convert_to_numpy(attn):
    return np.array([layer.to('cpu').squeeze().detach().numpy() for layer in attn])

def calc_attention_rollout(attn):
    full_attn = 0.5*attn+0.5*np.eye(attn.shape[-1])[None,None,...]
    ave_attn = full_attn.mean(axis=1)
    attn_rollout = [full_attn[0]]
    ave_attn_rollout = [ave_attn[0]]
    for layer,ave_layer in zip(full_attn[1:],ave_attn[1:]):
        layer_rollout = np.array([list(head@ave_attn_rollout[-1]) for head in layer])
        attn_rollout.append(layer_rollout)
        ave_attn_rollout.append(ave_layer@ave_attn_rollout[-1])
    return np.array(attn_rollout)

def CalcAttn(head,line,sent_id):
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
        if args.roll_out:
            attention = calc_attention_rollout(attention)
            attention_masked = calc_attention_rollout(attention_masked)

    attn_choice_correct = attention[:,:,pron_token_id,choice_start_id_correct:choice_end_id_correct].sum(axis=-1)
    attn_choice_incorrect = attention[:,:,pron_token_id,choice_start_id_incorrect:choice_end_id_incorrect].sum(axis=-1)
    attn_masked_choice_correct = attention_masked[:,:,pron_token_id,choice_start_id_correct:choice_end_id_correct].sum(axis=-1)
    attn_masked_choice_incorrect = attention_masked[:,:,pron_token_id,choice_start_id_incorrect:choice_end_id_incorrect].sum(axis=-1)
    attn_context_pron = attention[:,:,pron_token_id,context_start_id:context_end_id].sum(axis=-1)
    attn_context_all = attention[:,:,:,context_start_id:context_end_id].sum(axis=-1)
    attn_context_mean = attn_context_all.mean(axis=-1)
    attn_masked_context_pron = attention_masked[:,:,pron_token_id,context_start_id:context_end_id].sum(axis=-1)
    attn_masked_context_all = attention_masked[:,:,:,context_start_id:context_end_id].sum(axis=-1)
    attn_masked_context_mean = attn_masked_context_all.mean(axis=-1)
    attn_period = attention[:,:,pron_token_id,period_id]
    attn_masked_period = attention_masked[:,:,pron_token_id,period_id]

    attn_context_dict = {}
    attn_context_dict['pron'] = attn_context_pron
    attn_context_dict['mean'] = attn_context_mean
    attn_masked_context_dict = {}
    attn_masked_context_dict['pron'] = attn_masked_context_pron
    attn_masked_context_dict['mean'] = attn_masked_context_mean
    return np.array([attn_choice_correct,attn_choice_incorrect]),np.array([attn_masked_choice_correct,attn_masked_choice_incorrect]),attn_context_dict,attn_masked_context_dict,attn_period,attn_masked_period


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--stimuli', type = str,
                        choices=['original','control_gender','control_number','control_combined'],
                        default='original')
    parser.add_argument('--core_id', type = int, default=0)
    parser.add_argument('--roll_out',dest='roll_out',action='store_true')
    parser.set_defaults(roll_out=False)
    args = parser.parse_args()
    print(f'running with {args}')

    if args.roll_out:
        roll_out_id = '_roll_out'
    else:
        roll_out_id = ''

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

    out_dict = {}
    for line in text:
        #if 'willow-towered Canopy Huntertropic wrestles' in [line[head.index('choice_correct_1')],line[head.index('choice_correct_2')]]:
        #    print('passed 1')
        #    continue
        #elif 'My great-grandfather' in [line[head.index('choice_correct_1')],line[head.index('choice_correct_2')]]:
        #    print('passed 2')
        #    continue

        attn_choice_1,attn_choice_masked_1,attn_context_dict_1,attn_masked_context_dict_1,attn_period_1,attn_masked_period_1 = CalcAttn(head,line,1)
        attn_choice_2,attn_choice_masked_2,attn_context_dict_2,attn_masked_context_dict_2,attn_period_2,attn_masked_period_2 = CalcAttn(head,line,2)

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

    with open(f'datafile/superglue_wsc_attention_{args.model}_{args.stimuli}{roll_out_id}.pkl','wb') as f:
        pickle.dump(out_dict,f)
