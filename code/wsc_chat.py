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

def calc_continuation(prompt):
    pass_flag = False
    num_fails = 0
    while not pass_flag and num_fails<5:
        try:
            response = openai.ChatCompletion.create(
                model      = args.model,
                messages   = [
                        {"role": "system", "content": "You are a helpful assistant that can correctly answer which noun phrase a pronoun refers to based on the context."},
                        {"role": "user", "content": prompt}
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
    parser.add_argument('--model', type = str, required = True, choices = ['gpt-4-0314','gpt-3.5-turbo-0301','text-davinci-003'])
    parser.add_argument('--dataset', type = str, choices=['combined'], default = 'combined')
    parser.add_argument('--stimuli', type = str, choices=['prompt'], default='prompt')
    parser.add_argument('--size', type = str, choices=['xs','s','m','l','xl','debiased'])
    parser.add_argument('--core_id', type = int, default=0)
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
    dataset_name = args.dataset + f'_{args.size}' if args.dataset == 'winogrande' else args.dataset
    out_file_name = f'{os.environ.get("MY_DATA_PATH")}/chat/'+\
                    f'{dataset_name}_{args.stimuli}{mask_context_id}_prediction_{args.model}'

    start = time.time()
    os.makedirs(f'{os.environ.get("MY_DATA_PATH")}/chat/',exist_ok=True)
    with open(f'{out_file_name}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(head+['response','acc'])
        for line in text[:10]:
            logprobs = {}
            sent = line[head.index('sentence')]
            query = line[head.index('query')]
            option_1 = line[head.index('responseA')]
            option_2 = line[head.index('responseB')]
            correct = 'A' if line[head.index('1or2')]=='1' else 'B'
            prompt = f"{sent} {query.replace('?',',')} A: {option_1} or B: {option_2}?"
            print(prompt)
            response = calc_continuation(prompt)
            writer.writerow(line+[response, correct in response])
    print(f'{len(text)} sentences done')
    print(f'Time: {time.time()-start}')
