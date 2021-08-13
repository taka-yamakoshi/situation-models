import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import csv
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    with open('WSC/concat.jsonl','r') as f:
        file = f.readlines()
    loaded_data = [json.loads(line) for line in file][0]

    # Group together schema with the same sentences
    wsc_data = {}
    for schema in loaded_data:
        sent = schema['text']
        if sent not in wsc_data:
            wsc_data[sent] = []
        schema_data = {}
        schema_data['sent'] = sent
        schema_data['choice'] = schema['target']['span1_text']
        schema_data['pron'] = schema['target']['span2_text']
        schema_data['choice_word_id'] = int(schema['target']['span1_index'])
        schema_data['pron_word_id'] = int(schema['target']['span2_index'])
        schema_data['label'] = bool(schema['label'])
        wsc_data[sent].append(schema_data)

    # Select schema with exactly two sentences
    wsc_data_new = {}
    for key,value in wsc_data.items():
        if len(value)==2 and value[0]['label']^value[1]['label']:
            wsc_data_new[key] = value
            #if len(value[0]['choice'].split(' '))!=len(value[1]['choice'].split(' ')):
            #    print(key)
    print(f'{len(list(wsc_data_new.keys()))} sentences extracted')

    out_dict = {}
    for key,value in wsc_data_new.items():
        #key = 'The city councilmen refused the demonstrators a permit because they feared violence.'
        #value = wsc_data_new[key]
        schema_data = {}
        assert len(value)==2
        sent_1 = value[0]['sent']
        sent_2 = value[1]['sent']
        assert sent_1==sent_2
        sent = sent_1
        schema_data['sent'] = sent
        schema_data['split_sent'] = sent.split(' ')

        pron_1 = value[0]['pron']
        pron_2 = value[1]['pron']
        try:
            assert pron_1==pron_2
        except AssertionError:
            if sent=="Madonna fired her trainer because she couldn't stand her boyfriend.":
                print('passed 1')
                continue
            else:
                print('Something is wrong with the pronoun')
                exit()
        pron = pron_1
        schema_data['pron'] = pron

        assert value[0]['pron_word_id']==value[1]['pron_word_id']
        schema_data['pron_word_id'] = value[0]['pron_word_id']

        labels = [value[0]['label'],value[1]['label']]
        correct_id = [i for i,label in zip([0,1],labels) if label][0]

        schema_data['choice_correct'] = value[correct_id]['choice']
        schema_data['choice_incorrect'] = value[1-correct_id]['choice']
        schema_data['choice_word_id_correct'] = value[correct_id]['choice_word_id']
        schema_data['choice_word_id_incorrect'] = value[1-correct_id]['choice_word_id']

        # Apply heuristics to find the sentence pairs
        sent_key = ' '.join(sent.split(' ')[:5])
        if sent_key not in out_dict:
            out_dict[sent_key] = []
        out_dict[sent_key].append(schema_data)

    paired_sents = []
    single_sents = []
    others = []
    for key,value in out_dict.items():
        if len(value)==2:
            paired_sents.append(value)
        elif len(value)==1:
            single_sents.append(value)
        else:
            others.append(value)

    with open(f'datafile/SuperGLUE_wsc_pairs.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['pair_id','sent_1','sent_2','split_sent_1','split_sent_2',
        'pron','pron_word_id_1','pron_word_id_2',
        'choice_correct_1','choice_incorrect_1','choice_word_id_correct_1','choice_word_id_incorrect_1',
        'choice_correct_2','choice_incorrect_2','choice_word_id_correct_2','choice_word_id_incorrect_2'])
        for pair_id,line in enumerate(paired_sents):
            try:
                assert line[0]['pron']==line[1]['pron'],line
            except AssertionError:
                if "Mark was close to Mr. Singer 's heels. He heard him calling for the captain" in line[0]['sent']:
                    print('passed 2')
                    continue
            pron = line[0]['pron']
            writer.writerow([pair_id,line[0]['sent'],line[1]['sent'],line[0]['split_sent'],line[1]['split_sent'],
            pron,line[0]['pron_word_id'],line[1]['pron_word_id'],
            line[0]['choice_correct'],line[0]['choice_incorrect'],
            line[0]['choice_word_id_correct'],line[0]['choice_word_id_incorrect'],
            line[1]['choice_correct'],line[1]['choice_incorrect'],
            line[1]['choice_word_id_correct'],line[1]['choice_word_id_incorrect']])

    with open(f'datafile/SuperGLUE_wsc_singles.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['sent_id','sent','split_sent',
        'pron','pron_word_id',
        'choice_correct','choice_incorrect','choice_word_id_correct','choice_word_id_incorrect'])
        for sent_id,line in enumerate(single_sents):
            writer.writerow([sent_id,line[0]['sent'],line[0]['split_sent'],
            line[0]['pron'],line[0]['pron_word_id'],
            line[0]['choice_correct'],line[0]['choice_incorrect'],
            line[0]['choice_word_id_correct'],line[0]['choice_word_id_incorrect']])

    with open(f'datafile/SuperGLUE_wsc_others.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['sent_id','sent','split_sent',
        'pron','pron_word_id',
        'choice_correct','choice_incorrect','choice_word_id_correct','choice_word_id_incorrect'])
        for batch_id,batch in enumerate(others):
            for sent_id,line in enumerate(batch):
                writer.writerow([f'{batch_id}_{sent_id}',line['sent'],line['split_sent'],
                line['pron'],line['pron_word_id'],
                line['choice_correct'],line['choice_incorrect'],
                line['choice_word_id_correct'],line['choice_word_id_incorrect']])
