import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import argparse
import json
import csv
from SuperGLUE_WSC_alignment import AlignPronoun,AlignCandidates

def CalcProb(head,line,sent_id):
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

    choice_tokens_list = [input_sent[0][choice_start_id_correct:choice_end_id_correct],
    input_sent[0][choice_start_id_incorrect:choice_end_id_incorrect]]

    masked_sent_correct = input_sent.clone()
    masked_sent_incorrect = input_sent.clone()
    masked_sent_correct = torch.cat([masked_sent_correct[0][:pron_token_id],
                               torch.tensor([mask_id for token in choice_tokens_list[0]]),
                               masked_sent_correct[0][(pron_token_id+1):]]).unsqueeze(0)
    masked_sent_incorrect = torch.cat([masked_sent_incorrect[0][:pron_token_id],
                               torch.tensor([mask_id for token in choice_tokens_list[1]]),
                               masked_sent_incorrect[0][(pron_token_id+1):]]).unsqueeze(0)

    with torch.no_grad():
        outputs_correct = model(masked_sent_correct.to(args.device))
        outputs_incorrect = model(masked_sent_incorrect.to(args.device))
        probs_correct = F.log_softmax(outputs_correct[0][:, pron_token_id:(pron_token_id+len(choice_tokens_list[0]))], dim = -1).to('cpu')
        probs_incorrect = F.log_softmax(outputs_incorrect[0][:, pron_token_id:(pron_token_id+len(choice_tokens_list[1]))], dim = -1).to('cpu')

    choice_probs_sum = [np.sum([probs_correct[0,token_id,token].item() for token_id,token in enumerate(choice_tokens_list[0])]),
                        np.sum([probs_incorrect[0,token_id,token].item() for token_id,token in enumerate(choice_tokens_list[1])])]
    choice_probs_ave = [np.mean([probs_correct[0,token_id,token].item() for token_id,token in enumerate(choice_tokens_list[0])]),
                        np.mean([probs_incorrect[0,token_id,token].item() for token_id,token in enumerate(choice_tokens_list[1])])]

    return np.array(choice_probs_sum),np.array(choice_probs_ave)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--stimuli', type = str,
                        choices=['original','control_gender','control_number','control_combined'],
                        default='original')
    parser.add_argument('--core_id', type = int, default=0)
    args = parser.parse_args()
    print(f'running with {args}')

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

        choice_probs_sum_1,choice_probs_ave_1 = CalcProb(head,line,1)
        choice_probs_sum_2,choice_probs_ave_2 = CalcProb(head,line,2)

        out_dict[line[head.index('pair_id')]] = {}
        out_dict[line[head.index('pair_id')]]['sum_1'] = choice_probs_sum_1
        out_dict[line[head.index('pair_id')]]['sum_2'] = choice_probs_sum_2
        out_dict[line[head.index('pair_id')]]['ave_1'] = choice_probs_ave_1
        out_dict[line[head.index('pair_id')]]['ave_2'] = choice_probs_ave_2

    with open(f'datafile/superglue_wsc_prediction_{args.model}_{args.stimuli}.pkl','wb') as f:
        pickle.dump(out_dict,f)
