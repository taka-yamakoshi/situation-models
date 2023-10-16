# export MY_DATA_PATH='YOUR PATH TO DATA FILES'
import time
import numpy as np
import csv
import argparse
import os
import re

import torch
import torch.nn.functional as F

import openai
from dotenv import dotenv_values

from wsc_utils import load_dataset, load_model

def calc_continuation(user_prompt, chain_of_thought):
    if chain_of_thought:
        system_input = f"You are going to be shown a sentence and asked to fill in a blank (with an explanation). For example, consider the sentence, '"\
            +f"The man tried to put the tuba in a suitcase but <blank> was too small.'"\
            +f"We will ask whether the <blank> should be interpreted as the TUBA or the SUITCASE."\
            +f"The <blank> will be literally ambiguous, but you can use knowledge about the world to figure it out."\
            +f"In this case, we can reason as follows. The suitcase is a container, the man is attempting to fit a tuba in that container, and bigger things don't fit when containers are too small. In addition, tubas are generally bigger than suitcases. Therefore the answer is SUITCASE."\
            +f"What about the following sentence?"
    else :
        system_input = f"You are going to be shown a sentence and asked to fill in a blank. For example, consider the sentence, '"\
            +f"The man tried to put the tuba in a suitcase but <blank> was too small.'"\
            +f"We will ask whether the <blank> should be interpreted as the TUBA or the SUITCASE."\
            +f"What about the following sentence?"
    pass_flag = False
    num_fails = 0
    while not pass_flag and num_fails<100:
        try:
            response = openai.ChatCompletion.create(
                model      = args.model,
                messages   = [
                    {"role": "system", "content": system_input},
                    {"role": "user", "content": user_prompt}
                ],
                temperature = 1,
                )
            pass_flag = True
        except:
            time.sleep(30)
            num_fails += 1

    return response['choices'][0]['message']['content']

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True, choices = ['gpt-4-0314','gpt-3.5-turbo-0301'])
    parser.add_argument('--chain_of_thought', dest='chain_of_thought', action='store_true')
    parser.add_argument('--dataset', type = str, choices=['combined'], default = 'combined')
    parser.add_argument('--stimuli', type = str, choices=['prompt'], default='prompt')
    parser.add_argument('--core_id', type = int, default=0)
    parser.add_argument('--num_samples', type = int, default=1)
    parser.add_argument('--mask_context',dest='mask_context',action='store_true')
    parser.add_argument('--no_mask',dest='no_mask',action='store_true')
    parser.set_defaults(mask_context=False,no_mask=False)
    args = parser.parse_args()
    print(f'running with {args}')

    # set openAI key in separate .env file w/ content
    # OPENAIKEY = yourkey
    openai.api_key = dotenv_values('../.env')['OPENAIKEY']

    head,text = load_dataset(args)

    mask_context_id = '_mask_context' if args.mask_context else ''
    out_file_name = f'../dataset/chat/{args.dataset}_{args.stimuli}{mask_context_id}_prediction_{args.model}'

    start = time.time()
    os.makedirs(f'../dataset/chat/',exist_ok=True)
    with open(f'{out_file_name}.csv', 'w', encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow(head+['sample_id','response','acc'])
        for line in text:
            sent = line[head.index('sentence')]
            option_1 = line[head.index('responseA')]
            option_2 = line[head.index('responseB')]
            query = "Should the <blank> in this sentence be interpreted as " + option_1 + " or " + option_2 + "? Let's think step by step before we give an answer."
            prompt = f"{sent} {query}"
            print(prompt)
            for sample_id in range(args.num_samples):
                response = calc_continuation(prompt, args.chain_of_thought)
                print(response)
                writer.writerow(line+[sample_id, response, option_1.lower() in response.lower()])
    print(f'{len(text)} sentences done')
    print(f'Time: {time.time()-start}')
