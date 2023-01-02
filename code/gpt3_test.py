import time
import numpy as np
import csv
import argparse
import os

import openai
from dotenv import dotenv_values

from wsc_utils import load_dataset

def calc_logprob(initialSequence, continuation):
    response = openai.Completion.create(
            engine      = "text-davinci-003",
            prompt      = initinitialSequence_seq + " " + continuation,
            max_tokens  = 0,
            temperature = 1,
            logprobs    = 0,
            echo        = True
        )

    text_offsets = response.choices[0]['logprobs']['text_offset']
    cutIndex = text_offsets.index(max(i for i in text_offsets if i < len(initialSequence))) + 1
    endIndex = response.usage.total_tokens
    answerTokens = response.choices[0]["logprobs"]["tokens"][cutIndex:endIndex]
    answerTokenLogProbs = response.choices[0]["logprobs"]["token_logprobs"][cutIndex:endIndex]

    assert len(answerTokens)==1 and answerTokens[0]==" "+continuation
    assert len(answerTokenLogProbs)==1
    return np.mean(answerTokenLogProbs)

def create_stimuli(head,line,sent_id):
    sent = line[head.index(f'sent_{sent_id+1}')]
    assert len([_ for _ in re.finditer('_',sent)])==1
    question = sent[sent.index('_'):].replace('_','Which')[:-1]+'?\n'
    option_1 = line[head.index('option_1')]
    option_2 = line[head.index('option_2')]
    options = f'A: {option_1}\nB: {option_2}\n'
    initialSequence = sent+'\n'+question+options+'Answer:'
    return initialSequence

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--dataset', type = str, required = True, choices=['superglue','winogrande','combined'])
    parser.add_argument('--stimuli', type = str,
                        choices=['original','control','synonym_1','synonym_2'],
                        #'original_verb','control_combined','control_combined_verb','synonym_verb'],
                        default='original')
    parser.add_argument('--size', type = str, choices=['xs','s','m','l','xl','debiased'])
    parser.add_argument('--mask_context',dest='mask_context',action='store_true')
    parser.add_argument('--no_mask',dest='no_mask',action='store_true')
    parser.set_defaults(mask_context=False,no_mask=False)
    args = parser.parse_args()
    print(f'running with {args}')
    assert args.model=='gpt3'

    # set openAI key in separate .env file w/ content
    # OPENAIKEY = yourkey
    openai.api_key = dotenv_values('../.env')['OPENAIKEY']

    head,text = load_dataset(args)

    mask_context_id = '_mask_context' if args.mask_context else ''
    dataset_name = args.dataset + f'_{args.size}' if args.dataset == 'winogrande' else args.dataset
    out_file_name = f'{os.environ.get("MY_DATA_PATH")}/prediction/'+\
                    f'{dataset_name}_{args.stimuli}{mask_context_id}_prediction_{args.model}'

    start = time.time()
    with open(f'{out_file_name}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(head+['ave_1','ave_2'])
        for line in text:
            logprobs = {}
            for sent_id in range(2):
                initialSequence = create_stimuli(head,line,sent_id)
                logprobs[f'ave_{sent_id+1}'] = calc_logprob(initialSequence,'A')-calc_logprob(initialSequence,'B')
            writer.writerow(line+[logprobs['ave_1'],logprobs['ave_2']])

    print(f'{len(text)} sentences done')
    print(f'Time: {time.time()-start}')
