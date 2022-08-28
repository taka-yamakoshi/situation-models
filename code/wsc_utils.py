import numpy as np
import torch
import torch.nn.functional as F
from wsc_alignment import align_tokens, check_alignment
from model_skeleton import skeleton_model
import csv
import math

def load_dataset(args):
    # load the csv file
    if args.dataset=='superglue':
        if args.stimuli=='original':
            fname = '../dataset/SuperGLUE/SuperGLUE_wsc_new.csv'
        elif args.stimuli=='original_verb':
            fname = '../dataset/SuperGLUE/SuperGLUE_wsc_verb_new.csv'
        #elif args.stimuli=='control_gender':
        #    fname = '../dataset/SuperGLUE/SuperGLUE_wsc_new_control_gender.csv'
        #elif args.stimuli=='control_number':
        #    fname = '../dataset/SuperGLUE/SuperGLUE_wsc_new_control_number.csv'
        #elif args.stimuli=='control_combined':
        #    fname = '../dataset/SuperGLUE/SuperGLUE_wsc_control_combined_new.csv'
        elif args.stimuli=='control_combined_verb':
            fname = '../dataset/SuperGLUE/SuperGLUE_wsc_control_combined_verb_new.csv'
        elif args.stimuli=='synonym_verb':
            fname = '../dataset/SuperGLUE/SuperGLUE_wsc_synonym_verb_new.csv'
    elif args.dataset=='winogrande':
        if args.stimuli=='original':
            fname = f'../dataset/Winogrande/winogrande_{args.size}.csv'
        elif args.stimuli=='original_verb':
            fname = f'../dataset/Winogrande/winogrande_{args.size}_verb.csv'
    elif args.dataset=='combined':
        fname = f'../dataset/combined/{args.stimuli}.csv'

    with open(fname,'r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
    head = file[0]
    text = file[1:]
    return head,text

def load_model(args):
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
            path_to_ckpt = f'{os.environ.get("MISTRAL_GPT2_CKPT_PATH")}/gpt2-medium/{args.run}-x{run_dict[args.run]}-checkpoint-400000'
        elif 'small' in args.model:
            run_dict = {'alias':21,'battlestar':49,'caprica':81,'darkmatter':343,'expanse':777}
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            path_to_ckpt = f'{os.environ.get("MISTRAL_GPT2_CKPT_PATH")}/gpt2-small/{args.run}-x{run_dict[args.run]}-checkpoint-400000'
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


    args.device = torch.device("cuda", index=int(args.core_id)) if torch.cuda.is_available() else torch.device("cpu")
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

def calc_outputs(head,line,sent_id,model,tokenizer,mask_id,args,mask_context=False,output_for_attn=False,use_skeleton=False):
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
    if 'verb' in args.stimuli:
        verb = line[head.index(f'verb_{sent_id}')]
        verb_word_id = int(line[head.index(f'verb_word_id_{sent_id}')])
    else:
        verb = None
        verb_word_id = None

    input_sent = tokenizer(sent,return_tensors='pt')['input_ids']
    pron_start_id,pron_end_id = align_tokens(args,'pron',tokenizer,sent,input_sent,pron,pron_word_id)
    option_1_start_id,option_1_end_id = align_tokens(args,'option',tokenizer,sent,input_sent,option_1,option_1_word_id)
    option_2_start_id,option_2_end_id = align_tokens(args,'option',tokenizer,sent,input_sent,option_2,option_2_word_id)
    context_start_id,context_end_id = align_tokens(args,'context',tokenizer,sent,input_sent,context,context_word_id)
    if 'verb' in args.stimuli:
        verb_start_id,verb_end_id = align_tokens(args,'verb',tokenizer,sent,input_sent,verb,verb_word_id)
    else:
        verb_start_id,verb_end_id = 0,0
    period_id = align_tokens(args,'period',tokenizer,sent,input_sent,None,None)

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
            if not use_skeleton:
                output = model(masked_sent.to(args.device))
            else:
                default_output = model(masked_sent.expand(args.num_heads,-1).to(args.device))
                output = skeleton_model(0,default_output[1][0],model,{},args)
                for default_hidden,hidden in zip(default_output[1],output[1]):
                    assert torch.all(default_hidden == hidden)
                for default_attn,attn in zip(default_output[2],output[2]):
                    assert torch.all(default_attn == attn)
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
            if not use_skeleton:
                outputs = [model(masked_sent.to(args.device)) for masked_sent in masked_sents]
            else:
                default_outputs = [model(masked_sent.expand(args.num_heads,-1).to(args.device)) for masked_sent in masked_sents]
                outputs = [skeleton_model(0,default_output[1][0],model,{},args) for default_output in default_outputs]
                for default_output,output in zip(default_outputs,outputs):
                    for default_hidden,hidden in zip(default_output[1],output[1]):
                        assert torch.all(default_hidden == hidden)
                    for default_attn,attn in zip(default_output[2],output[2]):
                        assert torch.all(default_attn == attn)

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
        aligned_token_ids['verb'] = np.array([verb_start_id,verb_end_id])
        aligned_token_ids['period'] = period_id

        output_token_ids = check_re_alignment(sent_id,tokenizer,mask_id,masked_sent,
                                            options,context,verb,aligned_token_ids,
                                            mask_context,context_length,mask_length,
                                            pron,args,output_for_attn)
        output_token_ids['pron_id'] = torch.tensor([pron_start_id]).to(args.device)

        return output, output_token_ids, option_tokens_list, masked_sent
    else:
        output_token_ids = {}
        for i in range(2):
            mask_length = len(option_tokens_list[i])
            shift = mask_length - pron_length
            masked_option = options[i]
            aligned_token_ids[f'masked_sent_{i+1}'] = {}
            aligned_token_ids[f'masked_sent_{i+1}']['option_1'] = option_ids[0]+shift*(pron_start_id<option_ids[0][0])
            aligned_token_ids[f'masked_sent_{i+1}']['option_2'] = option_ids[1]+shift*(pron_start_id<option_ids[1][0])
            aligned_token_ids[f'masked_sent_{i+1}']['context'] = np.array([context_start_id,context_end_id])+shift*(pron_start_id<context_start_id)
            aligned_token_ids[f'masked_sent_{i+1}']['masks'] = np.array([pron_start_id,pron_start_id+mask_length])
            aligned_token_ids[f'masked_sent_{i+1}']['verb'] = np.array([verb_start_id,verb_end_id])+shift*(pron_start_id<verb_start_id)
            aligned_token_ids[f'masked_sent_{i+1}']['period'] = period_id+shift

            output_token_ids[f'masked_sent_{i+1}'] = check_re_alignment(sent_id,tokenizer,mask_id,masked_sents[i],
                                                                        options,context,verb,aligned_token_ids[f'masked_sent_{i+1}'],
                                                                        mask_context,context_length,mask_length,
                                                                        masked_option,args,output_for_attn)
        output_token_ids['pron_id'] = torch.tensor([pron_start_id]).to(args.device)

        return outputs, output_token_ids, option_tokens_list, masked_sents

def check_re_alignment(sent_id,tokenizer,mask_id,masked_sent,options,context,verb,aligned_token_ids,mask_context,context_length,mask_length,masked_option,args,output_for_attn):
    check_alignment(args,'option',tokenizer,masked_sent,options[0],*aligned_token_ids['option_1'])
    check_alignment(args,'option',tokenizer,masked_sent,options[1],*aligned_token_ids['option_2'])
    if 'verb' in args.stimuli:
        check_alignment(args,'verb',tokenizer,masked_sent,verb,*aligned_token_ids['verb'])
    if mask_context:
        check_alignment(args,'context',tokenizer,masked_sent,
                        ' '.join([tokenizer.decode([mask_id]) for _ in range(context_length)]),
                        *aligned_token_ids['context'])
    else:
        check_alignment(args,'context',tokenizer,masked_sent,context,*aligned_token_ids['context'])
    if 'bert' in args.model:
        if args.no_mask:
            check_alignment(args,'masks',tokenizer,masked_sent,
                            masked_option,
                            *aligned_token_ids['masks'])
        else:
            check_alignment(args,'masks',tokenizer,masked_sent,
                            ' '.join([tokenizer.decode([mask_id]) for _ in range(mask_length)]),
                            *aligned_token_ids['masks'])
    elif 'gpt2' in args.model:
        check_alignment(args,'masks',tokenizer,masked_sent,
                        masked_option,
                        *aligned_token_ids['masks'])
    try:
        check_alignment(args,'period',tokenizer,masked_sent,None,aligned_token_ids['period'],None)
    except AssertionError:
        ## Context may be stuck together with the period
        assert mask_context
        check_alignment(args,'context',tokenizer,masked_sent,tokenizer.decode([mask_id]),
                        aligned_token_ids['period'],
                        aligned_token_ids['period']+1)

    output_token_ids = {}
    for feature in ['option_1','option_2','context','verb','masks']:
        output_token_ids[feature] = torch.tensor([i for i in range(*aligned_token_ids[feature])]).to(args.device)
    output_token_ids['period'] = torch.tensor([aligned_token_ids['period']]).to(args.device)
    output_token_ids['cls'] = torch.tensor([0]).to(args.device)
    output_token_ids['sep'] = torch.tensor([len(masked_sent[0])-1]).to(args.device)
    output_token_ids['options'] = torch.tensor([i for i in range(*aligned_token_ids['option_1'])]
                                                +[i for i in range(*aligned_token_ids['option_2'])]).to(args.device)
    if 'verb' in args.stimuli:
        feature_names = ['option_1','option_2','context','verb','masks','period','cls','sep']
    else:
        feature_names = ['option_1','option_2','context','masks','period','cls','sep']
    token_ids_all = [i for feature in feature_names for i in output_token_ids[feature]]
    assert len(token_ids_all)==len(set(token_ids_all))
    output_token_ids['rest'] = torch.tensor([i for i in range(len(masked_sent[0])) if i not in token_ids_all]).to(args.device)
    for i in range(len(masked_sent[0])):
        output_token_ids[f'token_{i}'] = torch.tensor([i]).to(args.device)
    output_token_ids['option_correct'] = output_token_ids[f'option_{sent_id}'].clone()
    output_token_ids['option_incorrect'] = output_token_ids[f'option_{3-sent_id}'].clone()
    return output_token_ids

def evaluate_predictions(logits_1,logits_2,pron_token_id,tokens_list,args):
    if 'bert' in args.model:
        probs_1 = F.log_softmax(logits_1[:, pron_token_id:(pron_token_id+len(tokens_list[0]))], dim = -1).to('cpu')
        probs_2 = F.log_softmax(logits_2[:, pron_token_id:(pron_token_id+len(tokens_list[1]))], dim = -1).to('cpu')
    elif 'gpt2' in args.model:
        probs_1 = F.log_softmax(logits_1[:,:-1], dim = -1).to('cpu')
        probs_2 = F.log_softmax(logits_2[:,:-1], dim = -1).to('cpu')
    assert probs_1.shape[1]==len(tokens_list[0]) and probs_2.shape[1]==len(tokens_list[1])
    choice_probs_sum = [np.sum([probs_1[:,token_id,token].detach().numpy()
                                for token_id,token in enumerate(tokens_list[0])],axis=0).squeeze(),
                        np.sum([probs_2[:,token_id,token].detach().numpy()
                                for token_id,token in enumerate(tokens_list[1])],axis=0).squeeze()]
    choice_probs_ave = [np.mean([probs_1[:,token_id,token].detach().numpy()
                                for token_id,token in enumerate(tokens_list[0])],axis=0).squeeze(),
                        np.mean([probs_2[:,token_id,token].detach().numpy()
                                for token_id,token in enumerate(tokens_list[1])],axis=0).squeeze()]
    return np.array(choice_probs_sum),np.array(choice_probs_ave)

def get_reps(outputs,token_ids,layer_id,head_id,pos_type,rep_type,args,context_id=None):
    assert pos_type in ['','option_1','option_2','option_correct','option_incorrect','context','verb','masks','period','cls','sep','other','options','rest'] or pos_type.startswith('token')
    assert rep_type in ['layer','key','query','value','attention','z_rep']
    num_heads = args.num_heads
    head_dim = args.head_dim
    if rep_type=='layer':
        layer_rep = outputs[1][layer_id]
        assert len(layer_rep.shape)==3
        vec = layer_rep[:,:,head_dim*head_id:head_dim*(head_id+1)][0,token_ids[f'{pos_type}']]
        return vec
    elif rep_type in ['key','query','value','z_rep']:
        if rep_type=='query':
            qry = outputs[3][0][layer_id]
            assert len(qry.shape)==4
            vec = qry[0,head_id,token_ids[f'{pos_type}']]
        elif rep_type=='key':
            key = outputs[3][1][layer_id]
            assert len(key.shape)==4
            vec = key[0,head_id,token_ids[f'{pos_type}']]
        elif rep_type=='value':
            val = outputs[3][2][layer_id]
            assert len(val.shape)==4
            vec = val[0,head_id,token_ids[f'{pos_type}']]
        elif rep_type=='z_rep':
            z_rep = outputs[4][layer_id]
            assert len(z_rep.shape)==3
            vec = z_rep[:,:,head_dim*head_id:head_dim*(head_id+1)][0,token_ids[f'{pos_type}']]
        return vec
    elif rep_type=='attention':
        mat = outputs[2][layer_id][0,head_id]
        assert mat.shape[0]==mat.shape[1]
        if args.intervention_type=='swap':
            return mat
        else:
            correct_option = ['option_1','option_2'][context_id-1]
            incorrect_option = ['option_2','option_1'][context_id-1]
            if args.intervention_type=='correct_option_attn':
                mat = fix_attn(mat,token_ids,correct_option,'masks',args)
            elif args.intervention_type=='incorrect_option_attn':
                mat = fix_attn(mat,token_ids,incorrect_option,'masks',args)
            elif args.intervention_type=='context_attn':
                mat = fix_attn(mat,token_ids,'context','masks',args)
            elif args.intervention_type=='other_attn':
                mat = fix_attn(mat,token_ids,'other','masks',args)
            elif args.intervention_type=='option_context_attn':
                mat = fix_attn(mat,token_ids,'context',correct_option,args)
            elif args.intervention_type=='option_masks_attn':
                mat = fix_attn(mat,token_ids,'masks',correct_option,args)
            elif args.intervention_type=='context_context_attn':
                mat = fix_attn(mat,token_ids,'context','masks',args)
                mat = fix_attn(mat,token_ids,'context',correct_option,args)
            elif args.intervention_type=='context_masks_attn':
                mat = fix_attn(mat,token_ids,'context','masks',args)
                mat = fix_attn(mat,token_ids,'masks',correct_option,args)
            elif args.intervention_type=='lesion_context_attn':
                mat = fix_attn(mat,token_ids,'context','masks',args,reverse=True)
            elif args.intervention_type=='lesion_attn':
                mat = torch.zeros(mat.size()).to(args.device)
            elif args.intervention_type=='scramble_masks_attn':
                mat = scramble_attn(mat,token_ids,'masks',args)
            else:
                raise NotImplementedError(f'invalid intervention type: {args.intervention_type}')
            return mat
    else:
        raise NotImplementedError(f'rep_type "{rep_type}" is not supported')

def fix_attn(mat,token_ids,in_pos,out_pos,args,reverse=False):
    if not args.test:
        if reverse:
            patch = torch.ones((len(token_ids[out_pos]),mat.shape[1])).to(args.device)/(mat.shape[1]-len(token_ids[in_pos]))
            patch[:,token_ids[in_pos]] = 0
        else:
            patch = torch.zeros((len(token_ids[out_pos]),mat.shape[1])).to(args.device)
            patch[:,token_ids[in_pos]] = 1/len(token_ids[in_pos])
        mat[token_ids[out_pos],:] = patch.clone()
    return mat

def scramble_attn(mat,token_ids,out_pos,args):
    if not args.test:
        rand_ids = np.random.permutation(mat.shape[1])
        patch = torch.tensor([[mat[out_pos_id][rand_id] for rand_id in rand_ids]
                            for out_pos_id in token_ids[out_pos]]).to(args.device)
        mat[token_ids[out_pos],:] = patch.clone()
    return mat

def extract_QKV(vecs,pos,token_ids):
    assert len(vecs.shape)==4
    vecs = vecs[:,:,token_ids[pos],:]
    assert len(vecs.shape)==4
    return vecs.to('cpu').numpy()

def evaluate_QKV(rep_type,result_1,result_2,head_id_1,head_id_2,args):
    effect_list_dist = []
    effect_list_cos = []
    for context_id in [1,2]:
        for masked_sent_id in [1,2]:
            condition_id = f'masked_sent_{masked_sent_id}_context_{context_id}'
            pair_condition_id = f'masked_sent_{masked_sent_id}_context_{3-context_id}'
            vec_1 = result_1[f'{rep_type}_{condition_id}'][head_id_1]
            vec_2 = result_2[f'{rep_type}_{pair_condition_id}'][head_id_2]
            assert len(vec_1.shape)==3 and len(vec_2.shape)==3
            effect_list_dist.append(np.linalg.norm(vec_1-vec_2,axis=-1).mean(axis=-1))
            effect_list_cos.append(np.divide(np.sum(vec_1*vec_2,axis=-1),
                                            np.linalg.norm(vec_1,axis=-1)*np.linalg.norm(vec_2,axis=-1)).mean(axis=-1))
    return np.array(effect_list_dist).mean(axis=0),np.array(effect_list_cos).mean(axis=0)
