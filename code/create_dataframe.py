# export MY_DATA_PATH='YOUR PATH TO DATA FILES'
import pickle
import numpy as np
import pandas as pd
import os
import time

def loaded_df_preprocess(file_path,show_last_mod=False,show_score=False,cols=None):
    if show_last_mod:
        file_stat = os.stat(file_path)
        print(time.ctime(file_stat.st_mtime),file_path)
    loaded_df = pd.read_csv(file_path)
    if show_score:
        print(loaded_df.loc[lambda d: (d['interv_type']=='original')&(d['layer_id']==0)&(d['head_id']==0)].score.mean())
    loaded_df = loaded_df.loc[lambda d:d['interv_type']=='interv']
    if cols is None:
        cols = ['pair_id','sent_1','sent_2','layer_id','head_id','original_score',
                'effect_ave','masks-option-diff_effect_47',
                'masks-qry-cos_effect_47','options-key-cos_effect_47']
    loaded_df = loaded_df[cols]
    return loaded_df

def set_up_args(rep_type):
    if rep_type == 'layer-query-key-value':
        pos_types = ['options','context','masks','verb','period','cls-sep','rest']
        cascade, multihead = False, True
        cue_type_list = ['context','verb','context_verb']
        choose_head_0 = True
    elif rep_type == 'z_rep_concat':
        pos_types = ['options','context','masks','verb','period','cls-sep','rest']
        cascade, multihead = True, True
        cue_type_list = ['context','verb','context_verb']
        choose_head_0 = True
    elif rep_type in ['z_rep_indiv','query','key','value']:
        pos_types = ['options','context','masks','verb','period','cls-sep','rest']
        cascade, multihead = False, False
        cue_type_list = ['context','verb']
        choose_head_0 = False
    elif rep_type == 'q_and_k':
        pos_types = ['options-masks','masks-options','rest-options','rest-masks','options-rest','masks-rest','context-rest','verb-rest']
        cascade, multihead = False, False
        cue_type_list = ['context','verb']
        choose_head_0 = False
    elif rep_type == 'attention':
        pos_types = ['None']
        cascade, multihead = False, False
        cue_type_list = ['context','verb']
        choose_head_0 = False
    else:
        raise NotImplementedError()
    return pos_types,cascade,multihead,cue_type_list,choose_head_0

def convert_cue_type_to_stim(cue_type):
    if cue_type == 'context':
        stimuli,mask_context = 'original',False
    elif cue_type == 'verb':
        stimuli,mask_context = 'control',True
    elif cue_type == 'context_verb':
        stimuli,mask_context = 'control',False
    else:
        raise NotImplementedError()
    return stimuli,mask_context

if __name__ =='__main__':
    dataset = 'combined'
    size = 'xl'
    metric = 'effect_ave'
    rep_types = ['layer-query-key-value','z_rep_concat','z_rep_indiv','query','key','value','attention']
    models = ['albert-xxlarge-v2']
    if len(models) == 1:
        model_id = models[0]
    else:
        model_id = 'models'

    dataset_name = dataset + f'_{size}' if dataset == 'winogrande' else dataset

    os.makedirs(f'{os.environ.get("MY_DATA_PATH")}/intervention/combined/',exist_ok=True)
    for rep_type in rep_types:
        print(f'Running {rep_type}\n')
        pos_types,cascade,multihead,cue_type_list,choose_head_0 = set_up_args(rep_type)
        if rep_type.startswith('z_rep'):
            rep_type = 'z_rep'
        df = pd.DataFrame([])
        for model in models:
            cascade_id = '_cascade' if cascade else ''
            multihead_id = '_multihead' if multihead else ''
            for cue_type in cue_type_list:
                stimuli,mask_context = convert_cue_type_to_stim(cue_type)
                mask_context_id = '_mask_context' if mask_context else ''
                for pos_type in pos_types:
                    file_name = f'{os.environ.get("MY_DATA_PATH")}/intervention/{dataset_name}_{stimuli}{mask_context_id}_intervention_swap_'\
                                    +f'{pos_type}_{rep_type}_{model}_layer_all_head_all{cascade_id}{multihead_id}.csv'
                    loaded_df = loaded_df_preprocess(file_name,show_last_mod=True,
                                                     cols=['pair_id','sent_1','sent_2','layer_id','head_id','original_score',metric,'original_1','original_2'])
                    if choose_head_0:
                        loaded_df = loaded_df.loc[lamba d: d['head_id']==0]
                    loaded_df = loaded_df.assign(model=model,cue_type=cue_type,pos_type=pos_type,rep_type=f'{rep_type}{cascade_id}{multihead_id}')
                    df = pd.concat([df,loaded_df])
        df.to_csv(f'{os.environ.get("MY_DATA_PATH")}/intervention/combined/{dataset}_{metric}_{rep_type}{cascade_id}{multihead_id}_{model_id}.csv',index=False)
        print('\n\n')
