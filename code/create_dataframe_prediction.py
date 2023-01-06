# export MY_DATA_PATH='YOUR PATH TO DATA FILES'
import pickle
import numpy as np
import pandas as pd
import os
import time

def loaded_df_preprocess(file_path,stimuli,show_last_mod=False,show_score=False):
    if show_last_mod:
        file_stat = os.stat(file_path)
        print(time.ctime(file_stat.st_mtime),file_path)
    loaded_df = pd.read_csv(file_path)
    if stimuli in ['synonym_1','synonym_2']:
        loaded_df = loaded_df.assign(score=lambda df:(df.ave_1>0)&(df.ave_2>0))
    else:
        loaded_df = loaded_df.assign(score=lambda df:(df.ave_1>0)&(df.ave_2<0))
    if show_score:
        print(f'{np.mean(loaded_df.score)}')
    return loaded_df

if __name__ =='__main__':
    dataset = 'combined'
    size = 'xl'

    dataset_name = dataset + f'_{size}' if dataset == 'winogrande' else dataset
    model_list = ['bert-base-uncased','bert-large-cased','roberta-base','roberta-large',
                    'albert-base-v2','albert-large-v2','albert-xlarge-v2','albert-xxlarge-v2','gpt2','gpt2-large','gpt3']
    stimuli_list = ['original','control','control','synonym_1','synonym_2']
    mask_context_list = [False,True,False,False,False]
    cue_type_list = ['context','verb','context+verb','synonym_1','synonym_2']

    os.makedirs(f'{os.environ.get("MY_DATA_PATH")}/prediction/combined/',exist_ok=True)
    df = pd.DataFrame([])
    for model in model_list:
        for stimuli,mask_context,cue_type in zip(stimuli_list,mask_context_list,cue_type_list):
            mask_context_id = '_mask_context' if mask_context else ''
            file_name = f'{os.environ.get("MY_DATA_PATH")}/prediction/'\
                        +f'{dataset_name}_{stimuli}{mask_context_id}_prediction_{model}.csv'
            loaded_df = loaded_df_preprocess(file_name,stimuli,show_last_mod=True,show_score=True)
            loaded_df = loaded_df.assign(cue_type=cue_type,model=model)
            df = pd.concat([df,loaded_df])
    df.to_csv(f'{os.environ.get("MY_DATA_PATH")}/prediction/combined/{dataset}.csv',index=False)
