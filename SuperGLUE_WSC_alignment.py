import numpy as np

def AlignPronoun(tokenizer,sent,input_sent,pron,pron_word_id):
    sent_before_pron = ' '.join(sent.split(' ')[:pron_word_id])
    tokens_before_pron = tokenizer(sent_before_pron)['input_ids']
    pron_token_id = len(tokens_before_pron)-1
    recreated_pron = tokenizer.decode([input_sent[0][pron_token_id]]).strip().lower()
    assert recreated_pron==pron.lower(), f'check the pronoun of "{sent}"'
    return pron_token_id

def AlignCandidates(tokenizer,sent,input_sent,choice,choice_word_id):
    sent_before_choice = ' '.join(sent.split(' ')[:choice_word_id])
    choice_start_id = len(tokenizer(sent_before_choice)['input_ids'])-1
    sent_until_choice = ' '.join(sent.split(' ')[:choice_word_id]+choice.split(' '))
    choice_end_id = len(tokenizer(sent_until_choice)['input_ids'])-1
    recreated_choice = tokenizer.decode(input_sent[0][choice_start_id:choice_end_id]).strip().lower()
    assert recreated_choice==choice.strip().lower(), f'check the candidates of "{sent}"'
    return choice_start_id,choice_end_id

def AlignContext(tokenizer,sent,input_sent,context,context_word_id,verbose=False):
    sent_before_context = ' '.join(sent.split(' ')[:context_word_id])
    context_start_id = len(tokenizer(sent_before_context)['input_ids'])-1
    sent_until_context = ' '.join(sent.split(' ')[:context_word_id]+context.split(' '))
    context_end_id = len(tokenizer(sent_until_context)['input_ids'])-1
    recreated_context = tokenizer.decode(input_sent[0][context_start_id:context_end_id]).strip().lower()
    if verbose:
        print(sent_before_context)
        print(sent_until_context)
        print(context,recreated_context)
    assert recreated_context==context.strip().lower(), f'check the context of "{sent}"'
    return context_start_id,context_end_id

def AlignPeriod(tokenizer,sent,input_sent):
    period_id = len(input_sent[0])-2
    assert tokenizer.decode(input_sent[0][period_id]) in ['.','".'],tokenizer.decode(input_sent[0][period_id])
    return period_id
