import numpy as np
import torch

def align_tokens(args, target_name, tokenizer, sent, input_sent, word, word_id, verbose=False):
    assert target_name in ['pron', 'option', 'context', 'verb', 'period', 'other']
    if target_name == 'period':
        # Adjustment for special tokens
        period_id = len(input_sent[0]) -  2 if 'bert' in args.model else len(input_sent[0]) -  1
        check_alignment(args, target_name, tokenizer, input_sent, word, period_id, None, verbose)
        return period_id
    else:
        sent_before_target = ' '.join(sent.split(' ')[:word_id])
        sent_until_target = ' '.join(sent.split(' ')[:word_id] + word.split(' '))
        # Adjustment for special tokens
        target_start_id = len(tokenizer(sent_before_target)['input_ids']) -  1 if 'bert' in args.model else len(tokenizer(sent_before_target)['input_ids'])
        target_end_id = len(tokenizer(sent_until_target)['input_ids']) -  1 if 'bert' in args.model else len(tokenizer(sent_until_target)['input_ids'])
        if verbose:
            print(sent_before_target)
            print(sent_until_target)
        target = word
        check_alignment(args, target_name, tokenizer, input_sent, target, target_start_id, target_end_id, verbose)
        return target_start_id, target_end_id

def check_alignment(args, target_name, tokenizer, input_sent, word, start_id, end_id, verbose=False):
    assert target_name in ['pron', 'masks', 'option', 'context', 'verb', 'period', 'other']
    if target_name == 'period':
        recreated_target = tokenizer.decode(input_sent[0][start_id])
        assert '.' in recreated_target or recreated_target == '<|endoftext|>', recreated_target
    else:
        tokenized_target = tokenizer(word, return_tensors='pt')['input_ids']
        try:
            # Adjustment for special tokens
            if 'bert' in args.model:
                assert torch.all(input_sent[0][start_id:end_id] == tokenized_target[0][1:-1])
            elif 'gpt2' in args.model:
                assert torch.all(input_sent[0][start_id:end_id] == tokenized_target[0][1:])
        except (RuntimeError, AssertionError):
            recreated_target = tokenizer.decode(input_sent[0][start_id:end_id])
            recreated_target = recreated_target.replace("' ", "'").replace(" '", "'").replace('" ', '"').replace(' "', '"').strip(' ,.;:').lower()
            if verbose:
                print(word, recreated_target)
            assert recreated_target in [word.strip().lower(), ''.join(word.split(' ')).strip().lower()], f'check the alignment of {target_name}: {recreated_target}!={word.strip().lower()}'
