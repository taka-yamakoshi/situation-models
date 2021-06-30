import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertModel
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    args = parser.parse_args()

    model = BertForMaskedLM.from_pretrained(args.model,output_hidden_states=True,output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model.eval()
    mask_id = tokenizer.encode("[MASK]")[1:-1][0]

    with open('datafile/wsc_data_new.pkl','rb') as f:
        wsc_data = pickle.load(f)

    schema_id = 0
    out_dict = {}
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

        pron_1 = value[0]['pron']
        pron_2 = value[1]['pron']
        assert pron_1==pron_2
        pron_token = tokenizer(pron_1)['input_ids'][1:-1]
        print(f'pronoun: {pron_1} as {pron_token}')
        assert len(pron_token)==1
        assert np.sum([token==pron_token[0] for token in input_1[0]])==1
        assert np.sum([token==pron_token[0] for token in input_2[0]])==1
        pron_id_1 = list(input_1[0]).index(pron_token[0])
        pron_id_2 = list(input_2[0]).index(pron_token[0])

        masked_sent_1 = input_1.clone()
        masked_sent_2 = input_2.clone()
        masked_sent_1[0][pron_id_1] = mask_id
        masked_sent_2[0][pron_id_2] = mask_id
        print(masked_sent_1)
        print(masked_sent_2)
        print([tokenizer.decode([token]) for token in masked_sent_1[0]])
        print([tokenizer.decode([token]) for token in masked_sent_2[0]])

        outputs_1 = model(masked_sent_1)[0]
        probs_1 = F.log_softmax(outputs_1[:, pron_id_1], dim = -1)
        outputs_2 = model(masked_sent_2)[0]
        probs_2 = F.log_softmax(outputs_2[:, pron_id_2], dim = -1)


        choices_1 = value[0]['choices']
        choices_2 = value[1]['choices']
        assert choices_1[0]==choices_2[0] and choices_1[1]==choices_2[1]
        choices = choices_1
        print('choices')
        print(choices)

        choice_tokens_list = [tokenizer(choice)['input_ids'][1:-1] for choice in choices]
        print(choice_tokens_list)

        choice_probs_sum_1 = [np.sum([probs_1[0,token].item() for token in tokens]) for tokens in choice_tokens_list]
        choice_probs_sum_2 = [np.sum([probs_2[0,token].item() for token in tokens]) for tokens in choice_tokens_list]
        choice_probs_ave_1 = [np.mean([probs_1[0,token].item() for token in tokens]) for tokens in choice_tokens_list]
        choice_probs_ave_2 = [np.mean([probs_2[0,token].item() for token in tokens]) for tokens in choice_tokens_list]

        correct_1 = value[0]['correct_ans']
        correct_2 = value[1]['correct_ans']
        print(correct_1)
        print(correct_2)

        correct_id_1 = ['A','B'].index(correct_1)
        correct_id_2 = ['A','B'].index(correct_2)
        print(correct_id_1,correct_id_2)
        out_dict[key] = {}
        out_dict[key]['sum'] = np.array([[choice_probs_sum_1[correct_id_1],choice_probs_sum_1[correct_id_2]],
        [choice_probs_sum_2[correct_id_2],choice_probs_sum_2[correct_id_1]]])
        out_dict[key]['ave'] = np.array([[choice_probs_ave_1[correct_id_1],choice_probs_ave_1[correct_id_2]],
        [choice_probs_ave_2[correct_id_2],choice_probs_ave_2[correct_id_1]]])

    with open(f'datafile/wsc_prediction_{args.model}.pkl','wb') as f:
        pickle.dump(out_dict,f)
