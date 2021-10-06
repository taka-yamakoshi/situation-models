import numpy as np

def AlignTokens(target_name,model_name,tokenizer,sent,input_sent,word,word_id,verbose=False):
    assert target_name in ['pron','choice','context','period']
    if target_name=='period':
        if 'bert' in model_name:
            period_id = len(input_sent[0])-2
        elif 'gpt2' in model_name:
            period_id = len(input_sent[0])-1
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
        if target_name=='pron':
            assert target_end_id==target_start_id+1
        if verbose:
            print(sent_before_target)
            print(sent_until_target)
        CheckAlignment(target_name,tokenizer,input_sent,word,target_start_id,target_end_id,verbose)
        return target_start_id,target_end_id

def CheckAlignment(target_name,tokenizer,input_sent,word,start_id,end_id,verbose=False):
    assert target_name in ['pron','choice','context','period']
    if target_name=='period':
        recreated_target = tokenizer.decode(input_sent[0][start_id])
        assert recreated_target in ['.','".','..','<|endoftext|>'],recreated_target
    else:
        recreated_target = tokenizer.decode(input_sent[0][start_id:end_id])
        if verbose:
            print(word,recreated_target)
        assert recreated_target.strip().lower()==word.strip().lower(), f'check the alignment of {target_name}'
