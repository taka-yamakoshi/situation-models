import time
import numpy as np
import csv
import argparse
import os
import re

import openai
from dotenv import dotenv_values

from wsc_utils import load_dataset, load_model

def calc_logprob_gpt(initialSequence, continuation, args):
    if args.model.startswith('gpt2'):
        model, tokenizer, mask_id, args = load_model(args)
        input_ids = tokenizer(initialSequence+" "+continuation,return_tensors='pt').input_ids
        assert tokenizer.decode(input_ids[0][-1]).strip()==continuation
        assert tokenizer.decode(input_ids[0][-2])==initialSequence[-1]
        with torch.no_grad():
            outputs = model(input_ids.to(args.device), return_dict=True)
        log_probs = F.log_softmax(outputs.logits.to('cpu'), dim = -1)
        assert log_probs.shape[0]==1
        return log_probs[0][-2][input_ids[0][-1].item()]
    else:
        assert args.model=='gpt3'
        pass_flag = False
        num_fails = 0
        while not pass_flag and num_fails<5:
            try:
                response = openai.Completion.create(
                    engine      = "text-davinci-003",
                    prompt      = initialSequence + " " + continuation,
                    max_tokens  = 0,
                    temperature = 1,
                    logprobs    = 0,
                    echo        = True
                    )
                pass_flag = True
            except:
                time.sleep(30)
                num_fails += 1

        text_offsets = response.choices[0]['logprobs']['text_offset']
        cutIndex = text_offsets.index(max(i for i in text_offsets if i < len(initialSequence))) + 1
        endIndex = response.usage.total_tokens
        answerTokens = response.choices[0]["logprobs"]["tokens"][cutIndex:endIndex]
        answerTokenLogProbs = response.choices[0]["logprobs"]["token_logprobs"][cutIndex:endIndex]

        assert len(answerTokens)==1 and answerTokens[0]==" "+continuation
        assert len(answerTokenLogProbs)==1
        return np.mean(answerTokenLogProbs)

def create_stimuli(head,line,sent_id):
    setUp = 'Situation: Sally went to the movies and Bob stayed home.\n'\
            +'Question: Which of them went to the movies, (A) Sally or (B) Bob?\n'\
            +'Answer: A\n\n'

    sent = line[head.index(f'sent_{sent_id+1}')]
    pron_new = line[head.index(f'pron_{sent_id+1}_new')]
    pron_replacement = line[head.index(f'pron_{sent_id+1}_replacement')]
    option_1 = line[head.index('option_1')]
    option_2 = line[head.index('option_2')]

    assert len([_ for _ in re.finditer('_',sent)])==1
    assert len([_ for _ in re.finditer(pron_new,sent)])==1

    sent_new = sent.replace(pron_new,pron_replacement)
    question = 'Which of them '+sent[sent.index('_')+2:][:-1]+', '
    options = f'(A) {option_1} or (B) {option_2}?\n'
    initialSequence = setUp+'Situation: '+sent_new+'\n'+'Question: '+question+options+'Answer:'
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

    if args.model=='gpt3':
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
            print(line)
            logprobs = {}
            for sent_id in range(2):
                initialSequence = create_stimuli(head,line,sent_id)
                logprob_A = calc_logprob_gpt(initialSequence,'A',args)
                logprob_B = calc_logprob_gpt(initialSequence,'B',args)
                logprobs[f'ave_{sent_id+1}'] = logprob_A - logprob_B
            writer.writerow(line+[logprobs['ave_1'],logprobs['ave_2']])

    print(f'{len(text)} sentences done')
    print(f'Time: {time.time()-start}')
