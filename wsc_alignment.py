import numpy as np

def AlignTokens(target_name,model_name,tokenizer,sent,input_sent,word,word_id,verbose=False):
    assert target_name in ['pron','choice','context','period']
    if target_name=='period':
        # if you didn't include <|endoftext|> token for gpt2 you need to fix here
        period_id = len(input_sent[0])-2
        CheckAlignment(target_name,tokenizer,input_sent,word,period_id,None,verbose)
        return period_id
    else:
        sent_before_target = ' '.join(sent.split(' ')[:word_id])
        sent_until_target = ' '.join(sent.split(' ')[:word_id]+word.split(' '))
        if 'bert' in model_name:
            target_start_id = len(tokenizer(sent_before_target)['input_ids'])-1
            target_end_id = len(tokenizer(sent_until_target)['input_ids'])-1
        elif 'gpt2' in model_name:
            target_start_id = len(tokenizer(sent_before_target)['input_ids'])
            target_end_id = len(tokenizer(sent_until_target)['input_ids'])
        if verbose:
            print(sent_before_target)
            print(sent_until_target)
        if target_name=='pron':
            if model_name.startswith('albert'):
                target = tokenizer.decode(tokenizer('_')['input_ids'])
            else:
                assert target_end_id==target_start_id+1
                target = word
        else:
            target = word
        CheckAlignment(target_name,tokenizer,input_sent,target,target_start_id,target_end_id,verbose)
        return target_start_id,target_end_id

def CheckAlignment(target_name,tokenizer,input_sent,word,start_id,end_id,verbose=False):
    assert target_name in ['pron','masks','choice','context','period']
    if target_name=='period':
        recreated_target = tokenizer.decode(input_sent[0][start_id])
        assert recreated_target in ['.','".','..','$20.'],recreated_target
    else:
        recreated_target = tokenizer.decode(input_sent[0][start_id:end_id])
        if verbose:
            print(word,recreated_target)
        if target_name=='masks':
            assert recreated_target.strip().lower() in [word.strip().lower(), ''.join(word.split(' ')).strip().lower()], f'check the alignment of {target_name}'
        else:
            assert recreated_target.strip().lower()==word.strip().lower(), f'check the alignment of {target_name}'
