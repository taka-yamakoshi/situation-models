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

if __name__ =='__main__':
    model = 'albert-xxlarge-v2'
    dataset = 'superglue'
    size = 'xl'
    metric = 'effect_ave'

    pos_types_singles = ['options','context','masks','verb','period','cls-sep','other','rest']
    pos_types_q_and_k = ['options-masks','masks-options','context-masks','context-options']
    cue_type_list = ['context','verb','context_verb','synonym']
    stimuli_list = ['original_verb','control_combined_verb','control_combined_verb','synonym_verb']
    mask_context_list = [False,True,False,False]
    rep_types = ['layer-query-key-value','z_rep','z_rep','value','q_and_k','attention']
    cascade_list = [False,True,False,False,False,False]
    multihead_list = [True,True,False,False,False,False]

    os.makedirs(f'{os.environ.get("MY_DATA_PATH")}/combined/',exist_ok=True)
    for rep_type,cascade,multihead in zip(rep_types,cascade_list,multihead_list):
        df = pd.DataFrame([])
        cascade_id = '_cascade' if cascade else ''
        multihead_id = '_multihead' if multihead else ''
        for stimuli,mask_context,cue_type in zip(stimuli_list,mask_context_list,cue_type_list):
            mask_context_id = '_mask_context' if mask_context else ''
            if rep_type=='attention':
                if dataset=='winogrande':
                    file_name = f'{os.environ.get("MY_DATA_PATH")}/winogrande_{size}{mask_context_id}_intervention_swap_'\
                                    +f'{rep_type}_{model}_layer_all_head_all{cascade_id}{multihead_id}.csv'
                elif dataset=='superglue':
                    file_name = f'{os.environ.get("MY_DATA_PATH")}/superglue_wsc_intervention_swap_'\
                                    +f'{rep_type}_{model}_{stimuli}{mask_context_id}_layer_all_head_all{cascade_id}{multihead_id}.csv'
                loaded_df = loaded_df_preprocess(file_name,show_last_mod=True,
                                                 cols=['pair_id','sent_1','sent_2','layer_id','head_id','original_score',metric])
                loaded_df = loaded_df.assign(cue_type=cue_type,pos_type='all',rep_type=f'{rep_type}{cascade_id}{multihead_id}')
                df = pd.concat([df,loaded_df])
            else:
                pos_types = pos_types_q_and_k if rep_type=='q_and_k' else pos_types_singles
                for pos_type in pos_types:
                    if dataset=='winogrande':
                        file_name = f'{os.environ.get("MY_DATA_PATH")}/winogrande_{size}{mask_context_id}_intervention_swap_'\
                                        +f'{pos_type}_{rep_type}_{model}_layer_all_head_all{cascade_id}{multihead_id}.csv'
                    elif dataset=='superglue':
                        file_name = f'{os.environ.get("MY_DATA_PATH")}/superglue_wsc_intervention_swap_'\
                                        +f'{pos_type}_{rep_type}_{model}_{stimuli}{mask_context_id}_layer_all_head_all{cascade_id}{multihead_id}.csv'
                    loaded_df = loaded_df_preprocess(file_name,show_last_mod=True,
                                                     cols=['pair_id','sent_1','sent_2','layer_id','head_id','original_score',metric])
                    loaded_df = loaded_df.assign(cue_type=cue_type,pos_type=pos_type,rep_type=f'{rep_type}{cascade_id}{multihead_id}')
                    df = pd.concat([df,loaded_df])
        df.to_csv(f'{os.environ.get("MY_DATA_PATH")}/combined/{dataset}_{metric}_{rep_type}{cascade_id}{multihead_id}.csv',index=False)