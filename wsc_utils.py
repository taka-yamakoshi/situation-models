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
        #elif args.stimuli=='control_gender':
        #    fname = 'datafile/SuperGLUE_wsc_new_control_gender.csv'
        #elif args.stimuli=='control_number':
        #    fname = 'datafile/SuperGLUE_wsc_new_control_number.csv'
        elif args.stimuli=='control_combined':
            fname = 'datafile/SuperGLUE_wsc_control_combined_new.csv'
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
        from transformers import RobertaTokenizer, RobertaForMaskedLM
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
    elif args.model.startswith('deberta'):
        from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM
        model = DebertaV2ForMaskedLM.from_pretrained(f'microsoft/{args.model}',output_hidden_states=True,output_attentions=True)
        tokenizer = DebertaV2Tokenizer.from_pretrained(f'microsoft/{args.model}')
    elif args.model.startswith('albert'):
        from transformers import AlbertTokenizer, AlbertForMaskedLM
        model = AlbertForMaskedLM.from_pretrained(args.model,output_hidden_states=True,output_attentions=True)
        tokenizer = AlbertTokenizer.from_pretrained(args.model)
    else:
        raise NotImplementedError("invalid model name")


    if torch.cuda.is_available():
        args.device = torch.device("cuda", index=int(args.core_id))
    else:
        args.device = torch.device("cpu")
    model.to(args.device)
    model.eval()
    if args.model.startswith('bert') or args.model.startswith('deberta') or args.model.startswith('albert'):
        mask_id = tokenizer.encode("[MASK]")[1:-1][0]
    elif args.model.startswith('roberta'):
        mask_id = tokenizer.encode("<mask>")[1:-1][0]
    elif 'gpt2' in args.model:
        mask_id = tokenizer.encode("_")[0]
    else:
        raise NotImplementedError("invalid model name")

    return model, tokenizer, mask_id, args

def CalcOutputs(head,line,sent_id,model,tokenizer,mask_id,args,mask_context=False,output_for_attn=False):
    # load data from a line in csv
    sent = line[head.index(f'sent_{sent_id}')]
    if 'gpt2' in args.model:
        sent += ' <|endoftext|>'
    pron = line[head.index(f'pron_{sent_id}')]
    pron_word_id = int(line[head.index(f'pron_word_id_{sent_id}')])
    option_1 = line[head.index(f'option_1')]
    option_2 = line[head.index(f'option_2')]
    option_1_word_id = int(line[head.index(f'option_1_word_id_{sent_id}')])
    option_2_word_id = int(line[head.index(f'option_2_word_id_{sent_id}')])
    context = line[head.index(f'context_{sent_id}')]
    context_word_id = int(line[head.index(f'context_word_id')])

    input_sent = tokenizer(sent,return_tensors='pt')['input_ids']
    pron_start_id,pron_end_id = AlignTokens(args,'pron',tokenizer,sent,input_sent,pron,pron_word_id)
    option_1_start_id,option_1_end_id = AlignTokens(args,'choice',tokenizer,sent,input_sent,option_1,option_1_word_id)
    option_2_start_id,option_2_end_id = AlignTokens(args,'choice',tokenizer,sent,input_sent,option_2,option_2_word_id)
    context_start_id,context_end_id = AlignTokens(args,'context',tokenizer,sent,input_sent,context,context_word_id)
    period_id = AlignTokens(args,'period',tokenizer,sent,input_sent,None,None)

    option_tokens_list = [input_sent[0][option_1_start_id:option_1_end_id],
                            input_sent[0][option_2_start_id:option_2_end_id]]
    option_ids = [np.array([option_1_start_id,option_1_end_id]),
                    np.array([option_2_start_id,option_2_end_id])]
    options = [option_1,option_2]

    if mask_context:
        input_context_masked = input_sent.clone()
        input_context_masked[0][context_start_id:context_end_id] = mask_id
        input_sent_new = input_context_masked.clone()
    else:
        input_sent_new = input_sent.clone()

    if output_for_attn:
        assert 'bert' in args.model
        masked_sent = input_sent_new.clone()
        if not args.no_mask:
            masked_sent[:,pron_start_id:pron_end_id] = mask_id
        with torch.no_grad():
            output = model(masked_sent.to(args.device))

    else:
        if 'bert' in args.model:
            masked_sents = [torch.cat([input_sent_new[0][:pron_start_id],
                                       torch.tensor([mask_id for token in option_tokens_list[option_id]]),
                                       input_sent_new[0][pron_end_id:]]).unsqueeze(0)
                            for option_id in range(2)]
        elif 'gpt2' in args.model:
            masked_sents = [torch.cat([input_sent_new[0][:pron_start_id],
                                       option_tokens_list[option_id].clone(),
                                       input_sent_new[0][pron_end_id:]]).unsqueeze(0)
                            for option_id in range(2)]

        with torch.no_grad():
            outputs = [model(masked_sent.to(args.device)) for masked_sent in masked_sents]

    # realign tokens
    aligned_token_ids = {}
    pron_length = pron_end_id-pron_start_id
    context_length = context_end_id-context_start_id
    if output_for_attn:
        assert 'bert' in args.model
        mask_length = pron_end_id-pron_start_id
        aligned_token_ids['option_1'] = option_ids[0]
        aligned_token_ids['option_2'] = option_ids[1]
        aligned_token_ids['context'] = np.array([context_start_id,context_end_id])
        aligned_token_ids['masks'] = np.array([pron_start_id,pron_end_id])
        aligned_token_ids['period'] = period_id

        output_token_ids = CheckRealignment(tokenizer,mask_id,masked_sent,
                                            options,context,aligned_token_ids,
                                            mask_context,context_length,mask_length,
                                            pron,args,output_for_attn)
        output_token_ids['pron_id'] = torch.tensor([pron_start_id]).to(args.device)

        return output, output_token_ids, option_tokens_list, masked_sent
    else:
        output_token_ids = {}
        for i in range(2):
            mask_length = len(option_tokens_list[i])
            masked_option = options[i]
            aligned_token_ids[f'masked_sent_{i+1}'] = {}
            aligned_token_ids[f'masked_sent_{i+1}']['option_1'] = option_ids[0]+(len(option_tokens_list[i])-pron_length)*(pron_start_id<option_ids[0][0])
            aligned_token_ids[f'masked_sent_{i+1}']['option_2'] = option_ids[1]+(len(option_tokens_list[i])-pron_length)*(pron_start_id<option_ids[1][0])
            aligned_token_ids[f'masked_sent_{i+1}']['context'] = np.array([context_start_id,context_end_id])+(len(option_tokens_list[i])-pron_length)*(pron_start_id<context_start_id)
            aligned_token_ids[f'masked_sent_{i+1}']['masks'] = np.array([pron_start_id,pron_start_id+len(option_tokens_list[i])])
            aligned_token_ids[f'masked_sent_{i+1}']['period'] = period_id+len(option_tokens_list[i])-pron_length

            output_token_ids[f'masked_sent_{i+1}'] = CheckRealignment(tokenizer,mask_id,masked_sents[i],
                                                                        options,context,aligned_token_ids[f'masked_sent_{i+1}'],
                                                                        mask_context,context_length,mask_length,
                                                                        masked_option,args,output_for_attn)
        output_token_ids['pron_id'] = torch.tensor([pron_start_id]).to(args.device)

        return outputs, output_token_ids, option_tokens_list, masked_sents

def CheckRealignment(tokenizer,mask_id,masked_sent,options,context,aligned_token_ids,mask_context,context_length,mask_length,masked_option,args,output_for_attn):
    CheckAlignment('choice',tokenizer,masked_sent,options[0],*aligned_token_ids['option_1'])
    CheckAlignment('choice',tokenizer,masked_sent,options[1],*aligned_token_ids['option_2'])
    if mask_context:
        CheckAlignment('context',tokenizer,masked_sent,
                        ' '.join([tokenizer.decode([mask_id]) for _ in range(context_length)]),
                        *aligned_token_ids['context'])
    else:
        CheckAlignment('context',tokenizer,masked_sent,context,*aligned_token_ids['context'])
    if 'bert' in args.model:
        if output_for_attn and args.no_mask:
            CheckAlignment('masks',tokenizer,masked_sent,
                            masked_option,
                            *aligned_token_ids['masks'])
        else:
            CheckAlignment('masks',tokenizer,masked_sent,
                            ' '.join([tokenizer.decode([mask_id]) for _ in range(mask_length)]),
                            *aligned_token_ids['masks'])
    elif 'gpt2' in args.model:
        CheckAlignment('masks',tokenizer,masked_sent,
                        masked_option,
                        *aligned_token_ids['masks'])
    try:
        CheckAlignment('period',tokenizer,masked_sent,None,aligned_token_ids['period'],None)
    except AssertionError:
        assert mask_context
        CheckAlignment('context',tokenizer,masked_sent,tokenizer.decode([mask_id]),
                        aligned_token_ids['period'],
                        aligned_token_ids['period']+1)

    output_token_ids = {}
    output_token_ids['option_1'] = torch.tensor([i for i in range(*aligned_token_ids['option_1'])]).to(args.device)
    output_token_ids['option_2'] = torch.tensor([i for i in range(*aligned_token_ids['option_2'])]).to(args.device)
    output_token_ids['context'] = torch.tensor([i for i in range(*aligned_token_ids['context'])]).to(args.device)
    output_token_ids['masks'] = torch.tensor([i for i in range(*aligned_token_ids['masks'])]).to(args.device)
    output_token_ids['period'] = torch.tensor([aligned_token_ids['period']]).to(args.device)
    output_token_ids['cls'] = torch.tensor([0]).to(args.device)
    output_token_ids['sep'] = torch.tensor([-1]).to(args.device)

    return output_token_ids

def EvaluatePredictions(logits_1,logits_2,token_ids,tokens_list,args):
    if 'bert' in args.model:
        pron_token_id = token_ids['pron_id']
        probs_1 = F.log_softmax(logits_1[:, pron_token_id:(pron_token_id+len(tokens_list[0]))], dim = -1).to('cpu')
        probs_2 = F.log_softmax(logits_2[:, pron_token_id:(pron_token_id+len(tokens_list[1]))], dim = -1).to('cpu')
    elif 'gpt2' in args.model:
        probs_1 = F.log_softmax(logits_1, dim = -1).to('cpu')
        probs_2 = F.log_softmax(logits_2, dim = -1).to('cpu')
    assert probs_1.shape[1]==len(tokens_list[0]) and probs_2.shape[1]==len(tokens_list[1])
    choice_probs_sum = [np.sum([probs_1[0,token_id,token].item() for token_id,token in enumerate(tokens_list[0])]),
                        np.sum([probs_2[0,token_id,token].item() for token_id,token in enumerate(tokens_list[1])])]
    choice_probs_ave = [np.mean([probs_1[0,token_id,token].item() for token_id,token in enumerate(tokens_list[0])]),
                        np.mean([probs_2[0,token_id,token].item() for token_id,token in enumerate(tokens_list[1])])]
    return np.array(choice_probs_sum),np.array(choice_probs_ave)
