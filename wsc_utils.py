import numpy as np
import torch
import torch.nn.functional as F
from wsc_alignment import AlignTokens, CheckAlignment
import csv

def LoadDataset(args):
    # load the csv file
    if args.dataset=='superglue':
        if args.stimuli=='original':
            fname = 'datafile/SuperGLUE_wsc_new.csv'
        elif args.stimuli=='control_gender':
            fname = 'datafile/SuperGLUE_wsc_new_control_gender.csv'
        elif args.stimuli=='control_number':
            fname = 'datafile/SuperGLUE_wsc_new_control_number.csv'
        elif args.stimuli=='control_combined':
            fname = 'datafile/SuperGLUE_wsc_new_control_combined.csv'
    elif args.dataset=='winogrande':
        fname = f'datafile/winogrande_{args.size}.csv'

    with open(fname,'r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
    head = file[0]
    text = file[1:]
    return head,text

def LoadModel(args):
    # load the model
    if args.model.startswith('bert'):
        from transformers import BertTokenizer, BertForMaskedLM
        model = BertForMaskedLM.from_pretrained(args.model,output_hidden_states=True,output_attentions=True)
        tokenizer = BertTokenizer.from_pretrained(args.model)
    elif args.model.startswith('roberta'):
        from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
        model = RobertaForMaskedLM.from_pretrained(args.model,output_hidden_states=True,output_attentions=True)
        tokenizer = RobertaTokenizer.from_pretrained(args.model)
    elif args.model.startswith('gpt2'):
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(args.model, output_hidden_states=True, output_attentions=True)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    elif args.model.startswith('mistral'):
        from mistral.src.models.mistral_gpt2 import MistralGPT2LMHeadModel
        from transformers import GPT2Tokenizer
        args.run = args.model.split('_')[-1]
        if 'medium' in args.model:
            run_dict = {'arwen':21,'beren':49,'celebrimbor':81,'durin':343,'eowyn':777}
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            path_to_ckpt = f"/jukebox/griffiths/situation_language/mistral_ckpt/gpt2-medium/{args.run}-x{run_dict[args.run]}-checkpoint-400000"
        elif 'small' in args.model:
            run_dict = {'alias':21,'battlestar':49,'caprica':81,'darkmatter':343,'expanse':777}
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            path_to_ckpt = f"/jukebox/griffiths/situation_language/mistral_ckpt/gpt2-small/{args.run}-x{run_dict[args.run]}-checkpoint-400000"
        model = MistralGPT2LMHeadModel.from_pretrained(path_to_ckpt)


    if torch.cuda.is_available():
        args.device = torch.device("cuda", index=int(args.core_id))
    else:
        args.device = torch.device("cpu")
    model.to(args.device)
    model.eval()
    if args.model.startswith('bert'):
        mask_id = tokenizer.encode("[MASK]")[1:-1][0]
    elif args.model.startswith('roberta'):
        mask_id = tokenizer.encode("<mask>")[1:-1][0]
    elif 'gpt2' in args.model:
        mask_id = tokenizer.encode("_")[0]

    return model, tokenizer, mask_id, args

def CalcOutputs(head,line,sent_id,model,tokenizer,mask_id,args,mask_context=False):
    # load data from a line in csv
    sent = line[head.index(f'sent_{sent_id}')]
    if 'gpt2' in args.model:
        sent += ' <|endoftext|>'
    pron = line[head.index(f'pron_{sent_id}')]
    pron_word_id = int(line[head.index(f'pron_word_id_{sent_id}')])
    choice_correct = line[head.index(f'choice_correct_{sent_id}')]
    choice_incorrect = line[head.index(f'choice_incorrect_{sent_id}')]
    choice_word_id_correct = int(line[head.index(f'choice_word_id_correct_{sent_id}')])
    choice_word_id_incorrect = int(line[head.index(f'choice_word_id_incorrect_{sent_id}')])
    context = line[head.index(f'context_{sent_id}')]
    context_word_id = int(line[head.index(f'context_word_id_{sent_id}')])

    input_sent = tokenizer(sent,return_tensors='pt')['input_ids']
    pron_token_id,_ = AlignTokens('pron',args.model,tokenizer,sent,input_sent,pron,pron_word_id)
    choice_start_id_correct,choice_end_id_correct = AlignTokens('choice',args.model,tokenizer,sent,input_sent,choice_correct,choice_word_id_correct)
    choice_start_id_incorrect,choice_end_id_incorrect = AlignTokens('choice',args.model,tokenizer,sent,input_sent,choice_incorrect,choice_word_id_incorrect)
    context_start_id,context_end_id = AlignTokens('context',args.model,tokenizer,sent,input_sent,context,context_word_id)
    period_id = AlignTokens('period',args.model,tokenizer,sent,input_sent,None,None)

    choice_tokens_list = [input_sent[0][choice_start_id_correct:choice_end_id_correct],
    input_sent[0][choice_start_id_incorrect:choice_end_id_incorrect]]
    choice_ids = {'correct':np.array([choice_start_id_correct,choice_end_id_correct]),
                'incorrect':np.array([choice_start_id_incorrect,choice_end_id_incorrect])}
    choices = [choice_correct,choice_incorrect]

    if mask_context:
        input_context_masked = input_sent.clone()
        input_context_masked[0][context_start_id:context_end_id] = mask_id
        input_sent_new = input_context_masked.clone()
    else:
        input_sent_new = input_sent.clone()

    if 'bert' in args.model:
        masked_sent_correct = torch.cat([input_sent_new[0][:pron_token_id],
                                   torch.tensor([mask_id for token in choice_tokens_list[0]]),
                                   input_sent_new[0][(pron_token_id+1):]]).unsqueeze(0)
        masked_sent_incorrect = torch.cat([input_sent_new[0][:pron_token_id],
                                   torch.tensor([mask_id for token in choice_tokens_list[1]]),
                                   input_sent_new[0][(pron_token_id+1):]]).unsqueeze(0)
    elif 'gpt2' in args.model:
        masked_sent_correct = torch.cat([input_sent_new[0][:pron_token_id],
                                   choice_tokens_list[0].clone(),
                                   input_sent_new[0][(pron_token_id+1):]]).unsqueeze(0)
        masked_sent_incorrect = torch.cat([input_sent_new[0][:pron_token_id],
                                   choice_tokens_list[1].clone(),
                                   input_sent_new[0][(pron_token_id+1):]]).unsqueeze(0)

    masked_sents_list = [masked_sent_correct[0], masked_sent_incorrect[0]]
    with torch.no_grad():
        outputs_correct = model(masked_sent_correct.to(args.device))
        outputs_incorrect = model(masked_sent_incorrect.to(args.device))

    # realign tokens
    if mask_context:
        output_token_ids = {}
        output_token_ids['pron_id'] = torch.tensor([pron_token_id]).to(args.device)
    else:
        # realign tokens
        aligned_token_ids = {}
        output_token_ids = {}
        for i,sent_type in enumerate(['correct','incorrect']):
            aligned_token_ids[f'{sent_type}_sent'] = {}
            aligned_token_ids[f'{sent_type}_sent']['choice_correct'] = choice_ids['correct']+(len(choice_tokens_list[i])-1)*(pron_token_id<choice_ids['correct'][0])
            aligned_token_ids[f'{sent_type}_sent']['choice_incorrect'] = choice_ids['incorrect']+(len(choice_tokens_list[i])-1)*(pron_token_id<choice_ids['incorrect'][0])
            aligned_token_ids[f'{sent_type}_sent']['context'] = np.array([context_start_id,context_end_id])+(len(choice_tokens_list[i])-1)*(pron_token_id<context_start_id)
            aligned_token_ids[f'{sent_type}_sent']['period'] = period_id + len(choice_tokens_list[i])-1
            CheckAlignment('choice',tokenizer,masked_sents_list[i].unsqueeze(0),choice_correct,*aligned_token_ids[f'{sent_type}_sent']['choice_correct'])
            CheckAlignment('choice',tokenizer,masked_sents_list[i].unsqueeze(0),choice_incorrect,*aligned_token_ids[f'{sent_type}_sent']['choice_incorrect'])
            CheckAlignment('context',tokenizer,masked_sents_list[i].unsqueeze(0),context,*aligned_token_ids[f'{sent_type}_sent']['context'])
            CheckAlignment('period',tokenizer,masked_sents_list[i].unsqueeze(0),None,aligned_token_ids[f'{sent_type}_sent']['period'],None)
            output_token_ids[f'{sent_type}_sent'] = {}
            output_token_ids[f'{sent_type}_sent']['choice_correct'] = torch.tensor([i for i in range(*aligned_token_ids[f'{sent_type}_sent']['choice_correct'])]).to(args.device)
            output_token_ids[f'{sent_type}_sent']['choice_incorrect'] = torch.tensor([i for i in range(*aligned_token_ids[f'{sent_type}_sent']['choice_incorrect'])]).to(args.device)
            output_token_ids[f'{sent_type}_sent']['context'] = torch.tensor([i for i in range(*aligned_token_ids[f'{sent_type}_sent']['context'])]).to(args.device)
            output_token_ids[f'{sent_type}_sent']['period'] = torch.tensor([aligned_token_ids[f'{sent_type}_sent']['period']]).to(args.device)
        output_token_ids['pron_id'] = torch.tensor([pron_token_id]).to(args.device)

        '''
        choice_correct_start_id_in_correct = choice_start_id_correct + (len(choice_tokens_list[0])-1)*(pron_token_id<choice_start_id_correct)
        choice_correct_end_id_in_correct = choice_end_id_correct + (len(choice_tokens_list[0])-1)*(pron_token_id<choice_start_id_correct)
        choice_correct_start_id_in_incorrect = choice_start_id_correct + (len(choice_tokens_list[1])-1)*(pron_token_id<choice_start_id_correct)
        choice_correct_end_id_in_incorrect = choice_end_id_correct + (len(choice_tokens_list[1])-1)*(pron_token_id<choice_start_id_correct)

        choice_incorrect_start_id_in_correct = choice_start_id_incorrect + (len(choice_tokens_list[0])-1)*(pron_token_id<choice_start_id_incorrect)
        choice_incorrect_end_id_in_correct = choice_end_id_incorrect + (len(choice_tokens_list[0])-1)*(pron_token_id<choice_start_id_incorrect)
        choice_incorrect_start_id_in_incorrect = choice_start_id_incorrect + (len(choice_tokens_list[1])-1)*(pron_token_id<choice_start_id_incorrect)
        choice_incorrect_end_id_in_incorrect = choice_end_id_incorrect + (len(choice_tokens_list[1])-1)*(pron_token_id<choice_start_id_incorrect)

        context_start_id_in_correct = context_start_id + (len(choice_tokens_list[0])-1)*(pron_token_id<context_start_id)
        context_end_id_in_correct = context_end_id + (len(choice_tokens_list[0])-1)*(pron_token_id<context_start_id)
        context_start_id_in_incorrect = context_start_id + (len(choice_tokens_list[1])-1)*(pron_token_id<context_start_id)
        context_end_id_in_incorrect = context_end_id + (len(choice_tokens_list[1])-1)*(pron_token_id<context_start_id)

        period_id_in_correct = period_id + len(choice_tokens_list[0])-1
        period_id_in_incorrect = period_id + len(choice_tokens_list[1])-1


        CheckAlignment('choice',tokenizer,masked_sent_correct,choice_correct,)
        CheckAlignment('choice',tokenizer,masked_sent_incorrect,choice_correct,choice_correct_start_id_in_incorrect,choice_correct_end_id_in_incorrect)
        CheckAlignment('choice',tokenizer,masked_sent_correct,choice_incorrect,choice_incorrect_start_id_in_correct,choice_incorrect_end_id_in_correct)
        CheckAlignment('choice',tokenizer,masked_sent_incorrect,choice_incorrect,choice_incorrect_start_id_in_incorrect,choice_incorrect_end_id_in_incorrect)

        CheckAlignment('context',tokenizer,masked_sent_correct,context,context_start_id_in_correct,context_end_id_in_correct)
        CheckAlignment('context',tokenizer,masked_sent_incorrect,context,context_start_id_in_incorrect,context_end_id_in_incorrect)
        CheckAlignment('period',tokenizer,masked_sent_correct,None,period_id_in_correct,None)
        CheckAlignment('period',tokenizer,masked_sent_incorrect,None,period_id_in_incorrect,None)

        '''
    return outputs_correct, outputs_incorrect, output_token_ids, choice_tokens_list, masked_sents_list

def EvaluatePredictions(logits_correct,logits_incorrect,token_ids,tokens_list,args):
    pron_token_id = token_ids['pron_id']
    if 'bert' in args.model:
        probs_correct = F.log_softmax(logits_correct[:, pron_token_id:(pron_token_id+len(tokens_list[0]))], dim = -1).to('cpu')
        probs_incorrect = F.log_softmax(logits_incorrect[:, pron_token_id:(pron_token_id+len(tokens_list[1]))], dim = -1).to('cpu')
    elif 'gpt2' in args.model:
        probs_correct = F.log_softmax(logits_correct, dim = -1).to('cpu')
        probs_incorrect = F.log_softmax(logits_incorrect, dim = -1).to('cpu')
    assert probs_correct.shape[1]==len(tokens_list[0]) and probs_incorrect.shape[1]==len(tokens_list[1])
    choice_probs_sum = [np.sum([probs_correct[0,token_id,token].item() for token_id,token in enumerate(tokens_list[0])]),
                        np.sum([probs_incorrect[0,token_id,token].item() for token_id,token in enumerate(tokens_list[1])])]
    choice_probs_ave = [np.mean([probs_correct[0,token_id,token].item() for token_id,token in enumerate(tokens_list[0])]),
                        np.mean([probs_incorrect[0,token_id,token].item() for token_id,token in enumerate(tokens_list[1])])]
    return np.array(choice_probs_sum),np.array(choice_probs_ave)
