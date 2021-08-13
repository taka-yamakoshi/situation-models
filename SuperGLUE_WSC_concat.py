import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import argparse
import json

if __name__=='__main__':
    with open('WSC/train.jsonl','r') as f:
        file = f.readlines()
    train_data = [json.loads(line) for line in file]
    print(len(train_data))
    with open('WSC/val.jsonl','r') as f:
        file = f.readlines()
    val_data = [json.loads(line) for line in file]
    print(len(val_data))
    concat_data = train_data+val_data
    print(len(concat_data))
    with open('WSC/concat.jsonl','w') as f:
        json.dump(concat_data,f)
