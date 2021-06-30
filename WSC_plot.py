from scipy.stats import pearsonr
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    args = parser.parse_args()

    with open('datafile/wsc_data_new.pkl','rb') as f:
        wsc_data = pickle.load(f)

    with open(f'datafile/wsc_prediction_{args.model}.pkl','rb') as f:
        pred_data = pickle.load(f)

    with open(f'datafile/wsc_attention_{args.model}.pkl','rb') as f:
        attn_data = pickle.load(f)

    print(f'# sentence pairs used: {len(list(wsc_data.keys()))}')

    sents = list(wsc_data.keys())
    probs = np.array([pred_data[key]['ave'] for key in sents])
    attention = np.array([attn_data[key] for key in sents])

    pred_score = np.array([[sent[0][0]>sent[0][1],sent[1][0]>sent[1][1]] for sent in probs])
    print(pred_score.shape)
    print(pred_score.mean())
    attn_score = np.array([[[[sent[0][0]>sent[0][1],sent[1][0]>sent[1][1]]
    for sent in attention[:,:,:,layer_id,head_id]]
    for head_id in range(12)]
    for layer_id in range(12)])
    print(attn_score.shape)
