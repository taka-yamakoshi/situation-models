import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import csv
import argparse
import itertools
import difflib
import spacy
import os

def FindWord(sent,phrase):
    split_sent = sent.split(' ')
    split_phrase = phrase.split(' ')
    # find each word in the phrase
    find_phrase = [[word_id for word_id, sent_word in enumerate(split_sent) if sent_word==phrase_word] for phrase_word in split_phrase]
    # consider all possible combinations
    candidates = np.array(list(itertools.product(*find_phrase)))
    # find ones that are in a sequence
    candidate_test = np.array([np.all(np.diff(candidate)==1) for candidate in candidates])
    if candidate_test.sum()==0:
        return 'no match'
    elif candidate_test.sum()>1:
        return 'multiple matches'
    else:
        # return the id of the first word
        return int(candidates[candidate_test][0][0])

def FindContext(sent_1,sent_2):
    split_sent_1 = sent_1.split(' ')
    split_sent_2 = sent_2.split(' ')
    context_words_1 = []
    context_words_2 = []
    for i,line in enumerate(difflib.unified_diff(split_sent_1,split_sent_2)):
        if i>2:
            if line.startswith('-'):
                context_words_1.append(line[1:])
            elif line.startswith('+'):
                context_words_2.append(line[1:])
    context_1 = ' '.join(context_words_1)
    context_2 = ' '.join(context_words_2)
    return context_1, context_2

def FindPOS(nlp,sent,word,word_id):
    sent_before_target = ' '.join(sent.split(' ')[:word_id])
    sent_until_target = ' '.join(sent.split(' ')[:word_id]+word.split(' '))
    target_start_id = len(nlp(sent_before_target))
    target_end_id = len(nlp(sent_until_target))
    doc = nlp(sent)
    assert word==doc[target_start_id:target_end_id].text
    pos_labels = [token.pos_ for token in doc][target_start_id:target_end_id]
    tag_labels = [token.tag_ for token in doc][target_start_id:target_end_id]
    return '+'.join([simplify_pos(pos_label,tag_label) for pos_label,tag_label in zip(pos_labels,tag_labels)])

def simplify_pos(pos_label,tag_label):
    if pos_label in ['SCONJ','CCONJ']:
        return 'CONJ'
    elif pos_label in ['ADV','ADJ']:
        return 'MOD'
    elif pos_label=='VERB' and tag_label in ['VBG','VBN']:
        return 'MOD'
    elif pos_label=='AUX':
        return 'VERB'
    elif pos_label=='PROPN':
        return 'NOUN'
    else:
        return pos_label

def FindVerb(nlp,sent,pron,pron_word_id):
    sent_before_target = ' '.join(sent.split(' ')[:pron_word_id])
    sent_until_target = ' '.join(sent.split(' ')[:pron_word_id]+pron.split(' '))
    target_start_id = len(nlp(sent_before_target))
    target_end_id = len(nlp(sent_until_target))
    assert target_end_id-target_start_id==1

    doc = nlp(sent)
    masked_token = doc[target_start_id]
    assert masked_token.text==pron
    head_token = masked_token.head

    if head_token.pos_=='ADJ':
        children_pos_list = [token.pos_ for token in head_token.children]
        children_ids = [token.i for token in head_token.children]
        if 'AUX' not in children_pos_list:
            verb_id = None
        else:
            verb_id = children_ids[children_pos_list.index('AUX')]
            assert doc[verb_id].pos_=='AUX'
    else:
        verb_id = head_token.i

    if verb_id is not None and doc[verb_id].pos_ not in ['VERB','AUX']:
        verb_id = None

    if verb_id is None or verb_id==0:
        return None, None
    else:
        sent_before_verb = doc[:verb_id].text
        return len(sent_before_verb.split(' ')),doc[verb_id]

if __name__=='__main__':
    np.random.seed(seed=2021)
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type = str, required = True, choices=['xs','s','m','l','xl','debiased'])
    parser.add_argument('--verb', dest='verb', action = 'store_true', default = False)
    args = parser.parse_args()

    print(f'Running with {args}')

    if args.verb:
        verb_id = '_verb'
    else:
        verb_id = ''

    nlp = spacy.load('en_core_web_lg')

    with open(f'winogrande_1.1/train_{args.size}.jsonl','r') as f:
        file = f.readlines()
    loaded_data = [json.loads(line) for line in file]

    # Group together schema with the same sentences
    wsc_data = {}
    for schema in loaded_data:
        qID = schema['qID']
        pair_id = qID.split('-')[0]
        if pair_id not in wsc_data:
            wsc_data[pair_id] = []

        schema_data = {}
        schema_data['qID'] = qID
        schema_data['sentence'] = ' '.join([word for word in schema['sentence'].split(' ') if len(word)>0])
        #if '-' in schema_data['sentence'] or '/' in schema_data['sentence']:
        #    continue

        schema_data['pron'] = '_'
        pron_word_id = FindWord(schema_data['sentence'],schema_data['pron'])

        option_1 = schema['option1']
        option_2 = schema['option2']
        option_1_word_id = FindWord(schema_data['sentence'],option_1)
        option_2_word_id = FindWord(schema_data['sentence'],option_2)

        if type(pron_word_id) is str or type(option_1_word_id) is str or type(option_2_word_id) is str:
            continue
        assert type(pron_word_id) is int and type(option_1_word_id) is int and type(option_2_word_id) is int

        schema_data['pron_word_id'] = pron_word_id
        # Make sure option_1 is always before option_2
        if option_1_word_id < option_2_word_id:
            schema_data['option_1'] = option_1
            schema_data['option_2'] = option_2
            schema_data['option_1_word_id'] = option_1_word_id
            schema_data['option_2_word_id'] = option_2_word_id
            schema_data['answer'] = schema['answer']

        elif option_1_word_id > option_2_word_id:
            schema_data['option_1'] = option_2
            schema_data['option_2'] = option_1
            schema_data['option_1_word_id'] = option_2_word_id
            schema_data['option_2_word_id'] = option_1_word_id
            if schema['answer']=="1":
                schema_data['answer'] = "2"
            elif schema['answer']=="2":
                schema_data['answer'] = "1"

        wsc_data[pair_id].append(schema_data)

    # Select schema with exactly two sentences
    os.makedirs('Winogrande/',exist_ok=True)
    with open(f'Winogrande/winogrande_{args.size}{verb_id}.csv','w') as f:
        writer = csv.writer(f)
        head = ['pair_id','sent_1','sent_2','pron_1','pron_2',
                'pron_word_id_1','pron_word_id_2',
                'option_1','option_2',
                'option_1_word_id_1','option_2_word_id_1',
                'option_1_word_id_2','option_2_word_id_2',
                'context_1','context_2','context_word_id',
                'other','other_word_id_1','other_word_id_2','context_pos']
        if args.verb:
            head += ['verb_1','verb_2','verb_word_id_1','verb_word_id_2','verb_pos','verb_tag']
        writer.writerow(head)
        for key,value in wsc_data.items():
            if len(value)==2:
                schema_data_all = {}
                schema_data_all['pair_id'] = key

                # schema_1 has the context where option_1 is correct
                if value[0]['answer']=="1" and value[1]['answer']=="2":
                    schema_1 = value[0]
                    schema_2 = value[1]
                elif value[0]['answer']=="2" and value[1]['answer']=="1":
                    schema_1 = value[1]
                    schema_2 = value[0]
                else:
                    #print('Invalid Answers')
                    #print(value[0]['sentence'])
                    #print(value[1]['sentence'])
                    continue

                assert schema_1['option_1']==schema_2['option_1']
                assert schema_1['option_2']==schema_2['option_2']

                schema_data_all['option_1'] = schema_1['option_1']
                schema_data_all['option_2'] = schema_1['option_2']

                for sent_id,schema_data in zip([1,2],[schema_1,schema_2]):
                    schema_data_all[f'sent_{sent_id}'] = schema_data['sentence']
                    schema_data_all[f'pron_{sent_id}'] = schema_data['pron']
                    schema_data_all[f'pron_word_id_{sent_id}'] = schema_data['pron_word_id']
                    schema_data_all[f'option_1_word_id_{sent_id}'] = schema_data['option_1_word_id']
                    schema_data_all[f'option_2_word_id_{sent_id}'] = schema_data['option_2_word_id']

                # Identify the context word
                context_1,context_2 = FindContext(schema_data_all['sent_1'],schema_data_all['sent_2'])
                #if len(context_1.split(' '))>5 or len(context_2.split(' '))>5:
                #    continue
                #if "'" in context_1 or "'" in context_2 or "_" in context_1 or "_" in context_2 or "’" in context_1 or "’" in context_2:
                #    continue
                context_word_id_1 = FindWord(schema_data_all['sent_1'],context_1)
                context_word_id_2 = FindWord(schema_data_all['sent_2'],context_2)

                if type(context_word_id_1) is str or type(context_word_id_2) is str:
                    continue
                assert type(context_word_id_1) is int and type(context_word_id_2) is int
                if context_word_id_1!=context_word_id_2:
                    continue

                schema_data_all['context_1'] = context_1.strip(' ,.;:')
                schema_data_all['context_2'] = context_2.strip(' ,.;:')
                schema_data_all['context_word_id'] = context_word_id_1

                context_1_pos = FindPOS(nlp,schema_data_all['sent_1'],
                                        schema_data_all['context_1'],schema_data_all['context_word_id'])
                context_2_pos = FindPOS(nlp,schema_data_all['sent_2'],
                                        schema_data_all['context_2'],schema_data_all['context_word_id'])
                matched_ids = difflib.SequenceMatcher(None,context_1_pos,context_2_pos).find_longest_match(0,len(context_1_pos),0,len(context_2_pos))
                matched_pos_1 = context_1_pos[matched_ids[0]:matched_ids[0]+matched_ids[2]]
                matched_pos_2 = context_2_pos[matched_ids[1]:matched_ids[1]+matched_ids[2]]
                assert matched_pos_1 == matched_pos_2
                # This is just a heuristic to make sure the identified pos is a valid one
                # TODO: find a more systematic way to identify shared POS
                shared_pos = '+'.join([pos_label for pos_label in matched_pos_1.split('+') if len(pos_label)>1])
                if shared_pos=='':
                    shared_pos = 'mismatch'
                    #print(context_1,context_2,context_1_pos,context_2_pos)
                schema_data_all['context_pos'] = shared_pos

                if args.verb:
                    verb_word_id_1,verb_1 = FindVerb(nlp,schema_data_all[f'sent_1'],
                                                        schema_data_all[f'pron_1'],
                                                        schema_data_all[f'pron_word_id_1'])
                    verb_word_id_2,verb_2 = FindVerb(nlp,schema_data_all[f'sent_2'],
                                                        schema_data_all[f'pron_2'],
                                                        schema_data_all[f'pron_word_id_2'])
                    if verb_1 is None or verb_2 is None:
                        continue

                    if schema_data_all['context_word_id'] in [verb_word_id_1,verb_word_id_2]:
                        continue

                    if not (verb_1.text==verb_2.text and verb_1.pos_==verb_2.pos_ and verb_1.tag_==verb_2.tag_):
                        continue

                    if verb_1.tag_ not in ['VBP','VBZ','BES','MD']:
                        continue

                    schema_data_all['verb_1'] = verb_1.text
                    schema_data_all['verb_2'] = verb_2.text
                    schema_data_all['verb_word_id_1'] = verb_word_id_1
                    schema_data_all['verb_word_id_2'] = verb_word_id_2
                    schema_data_all['verb_pos'] = verb_1.pos_
                    schema_data_all['verb_tag'] = verb_1.tag_

                # Sample the "other word"
                split_sent_list = []
                other_word_ids_list = []
                other_words_list = []
                for sent_id in [1,2]:
                    word_ids = [*[schema_data_all[f'pron_word_id_{sent_id}']+i for i in range(len(schema_data_all[f'pron_{sent_id}'].split(' ')))],
                                *[schema_data_all[f'option_1_word_id_{sent_id}']+i for i in range(len(schema_data_all[f'option_1'].split(' ')))],
                                *[schema_data_all[f'option_2_word_id_{sent_id}']+i for i in range(len(schema_data_all[f'option_2'].split(' ')))],
                                *[schema_data_all[f'context_word_id']+i for i in range(len(schema_data_all[f'context_{sent_id}'].split(' ')))],
                                len(schema_data_all[f'sent_{sent_id}'].split(' '))-1]
                    if args.verb:
                        word_ids += [schema_data_all[f'verb_word_id_{sent_id}']+i for i in range(len(schema_data_all[f'verb_{sent_id}'].split(' ')))]
                    split_sent = schema_data_all[f'sent_{sent_id}'].split(' ')
                    other_word_ids = [i for i in range(len(split_sent)) if i not in word_ids]
                    other_words = [split_sent[word_id] for word_id in other_word_ids]

                    split_sent_list.append(split_sent)
                    other_word_ids_list.append(other_word_ids)
                    other_words_list.append(other_words)

                assert len(other_words_list[0])==len(other_words_list[1])
                if not np.all([word_1==word_2 for word_1,word_2 in zip(other_words_list[0],other_words_list[1])]):
                    #print(other_words_1)
                    #print(other_words_2)
                    continue

                other_word = ''
                while other_word.strip(' ,.;:')=='':
                    rand_id = np.random.choice(len(other_words_list[0]))
                    other_word_id_1 = other_word_ids_list[0][rand_id]
                    other_word_id_2 = other_word_ids_list[1][rand_id]
                    assert split_sent_list[0][other_word_id_1]==split_sent_list[1][other_word_id_2]
                    other_word = split_sent_list[0][other_word_id_1].strip(' ,.;:')

                schema_data_all[f'other_word_id_1'] = other_word_id_1
                schema_data_all[f'other_word_id_2'] = other_word_id_2
                schema_data_all[f'other'] = other_word

                writer.writerow([schema_data_all[feature] for feature in head])