import csv
import numpy as np
import argparse


if __name__ == '__main__':
    np.random.seed(seed=2021)
    parser = argparse.ArgumentParser()
    parser.add_argument('--stimuli',type=str,
                        choices=['original_verb','control_combined_verb'],default='original')
    args = parser.parse_args()
    if args.stimuli=='original_verb':
        fname = 'SuperGLUE_wsc_verb'
    elif args.stimuli=='control_combined_verb':
        fname = 'SuperGLUE_wsc_control_combined_verb'

    with open(f'datafile/{fname}.csv','r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
        head = file[0]
        text = file[1:]

    new_head = ['pair_id','sent_1','sent_2','pron_1','pron_2',
                'pron_word_id_1','pron_word_id_2',
                'option_1','option_2',
                'option_1_word_id_1','option_2_word_id_1',
                'option_1_word_id_2','option_2_word_id_2',
                'context_1','context_2','context_word_id',
                'verb_1','verb_2','verb_word_id_1','verb_word_id_2',
                'other','other_word_id_1','other_word_id_2']

    with open(f'datafile/{fname}_new.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(new_head)
        for line in text:
            pron_1 = line[head.index('pron_1')]
            pron_2 = line[head.index('pron_2')]
            pron_word_id_1 = int(line[head.index('pron_word_id_1')])
            pron_word_id_2 = int(line[head.index('pron_word_id_2')])

            sent_1 = line[head.index('sent_1')]
            sent_2 = line[head.index('sent_2')]

            option_1 = line[head.index('option_1')]
            option_2 = line[head.index('option_2')]
            option_1_word_id_1 = int(line[head.index('option_1_word_id_1')])
            option_2_word_id_1 = int(line[head.index('option_2_word_id_1')])
            option_1_word_id_2 = int(line[head.index('option_1_word_id_2')])
            option_2_word_id_2 = int(line[head.index('option_2_word_id_2')])

            context_1 = line[head.index('context_1')]
            context_2 = line[head.index('context_2')]
            context_word_id = int(line[head.index('context_word_id')])

            verb_1 = line[head.index('verb_1')]
            verb_2 = line[head.index('verb_2')]
            verb_word_id_1 = int(line[head.index('verb_word_id_1')])
            verb_word_id_2 = int(line[head.index('verb_word_id_2')])

            word_ids_1 = [*[pron_word_id_1+i for i in range(len(pron_1.split(' ')))],\
                        *[option_1_word_id_1+i for i in range(len(option_1.split(' ')))],\
                        *[option_2_word_id_1+i for i in range(len(option_2.split(' ')))],\
                        *[context_word_id+i for i in range(len(context_1.split(' ')))],\
                        *[verb_word_id_1+i for i in range(len(verb_1.split(' ')))],\
                        len(sent_1.split(' '))-1]
            word_ids_2 = [*[pron_word_id_2+i for i in range(len(pron_2.split(' ')))],
                        *[option_1_word_id_2+i for i in range(len(option_1.split(' ')))],
                        *[option_2_word_id_2+i for i in range(len(option_2.split(' ')))],
                        *[context_word_id+i for i in range(len(context_2.split(' ')))],\
                        *[verb_word_id_2+i for i in range(len(verb_2.split(' ')))],\
                        len(sent_2.split(' '))-1]

            split_sent_1 = sent_1.split(' ')
            split_sent_2 = sent_2.split(' ')
            other_word_ids_1 = [i for i in range(len(split_sent_1)) if i not in word_ids_1]
            other_word_ids_2 = [i for i in range(len(split_sent_2)) if i not in word_ids_2]
            other_words_1 = [split_sent_1[word_id] for word_id in other_word_ids_1]
            other_words_2 = [split_sent_2[word_id] for word_id in other_word_ids_2]

            assert len(other_words_1)==len(other_words_2)
            if not np.all([word_1==word_2 for word_1,word_2 in zip(other_words_1,other_words_2)]):
                #print(other_words_1)
                #print(other_words_2)
                continue

            other_word = ''
            while other_word.strip(' ,.;:')=='':
                rand_id = np.random.choice(len(other_words_1))
                other_word_id_1 = other_word_ids_1[rand_id]
                other_word_id_2 = other_word_ids_2[rand_id]
                if split_sent_1[other_word_id_1]!=split_sent_2[other_word_id_2]:
                    continue
                else:
                    other_word = split_sent_1[other_word_id_1].strip(' ,.;:')

            writer.writerow([line[head.index('pair_id')],sent_1,sent_2,
                             pron_1,pron_2,pron_word_id_1,pron_word_id_2,
                             option_1,option_2,
                             option_1_word_id_1,option_2_word_id_1,
                             option_1_word_id_2,option_2_word_id_2,
                             context_1,context_2,context_word_id,
                             verb_1,verb_2,verb_word_id_1,verb_word_id_2,
                             other_word,other_word_id_1,other_word_id_2])
