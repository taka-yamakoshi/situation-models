import json
import numpy as np
import torch
from transformers import BertTokenizer
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pickle

if __name__=='__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load WSC sentences from the xml file
    tree = ET.parse('WSCollectionNew.xml')
    root = tree.getroot()

    # Group together sentences based on the prefix
    # TODO: there are cases where the context is in the prefix, and these are not grouped together
    wsc_data = {}
    for schema in root.iter('schema'):
        schema_dict = {}
        assert len(list(schema.iter('text')))==1
        for text in schema.iter('text'):
            assert len(list(text.iter('txt1')))==1
            assert len(list(text.iter('txt2')))==1
            assert len(list(text.iter('pron')))==1
            for txt1 in text.iter('txt1'):
                context = txt1.text.replace('\n',' ').strip()
            for txt2 in text.iter('txt2'):
                continuation = ' ' + txt2.text.replace('\n',' ').strip()
            for pron in text.iter('pron'):
                pronoun = ' ' + pron.text.replace('\n',' ').strip()

            if context not in wsc_data:
                wsc_data[context] = []

            schema_dict['sent'] = context+pronoun+continuation
            schema_dict['pron'] = pronoun.strip()
        assert len(list(schema.iter('answers')))==1
        for answers in schema.iter('answers'):
            answer_list = []
            assert len(list(answers.iter('answer')))==2
            for answer in answers.iter('answer'):
                answer_list.append(answer.text.replace('\n',' ').strip())
            schema_dict['choices'] = answer_list
        assert len(list(schema.iter('correctAnswer')))==1
        for correct_ans in schema.iter('correctAnswer'):
            schema_dict['correct_ans'] = correct_ans.text.replace('\n',' ').replace('.','').strip()
        wsc_data[context].append(schema_dict)

    with open('datafile/wsc_data.pkl','wb') as f:
        pickle.dump(wsc_data,f)

    schema_id = 0
    wsc_data_new = {}
    for key,value in wsc_data.items():
        if len(value)==2:
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
            print('pronoun:')
            print(pron_1,pron_2)
            print(pron_token)
            assert len(pron_token)==1
            try:
                assert np.sum([token==pron_token[0] for token in input_1[0]])==1
                assert np.sum([token==pron_token[0] for token in input_2[0]])==1
            except:
                print('There was an issue with pronouns')
                continue

            choices_1 = value[0]['choices']
            choices_2 = value[1]['choices']
            assert choices_1[0]==choices_2[0] and choices_1[1]==choices_2[1]
            choices = choices_1
            print('choices')
            print(choices)
            if choices[1] in ['Adam','Bob','Charlie','The mouse','chewing gum','Pete','My father','Yakutsk', 'Pam and Paul','Jade']:
                continue
            wsc_data_new[key] = value

    with open('datafile/wsc_data_new.pkl','wb') as f:
        pickle.dump(wsc_data_new,f)
