# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:59:19 2020

@author: zxr
"""
import pandas as pd
import numpy as np

import torch

def divideDataSet(dfw,dfk,test_ratio,indexK='isbn'):
    a = np.random.choice(len(dfk),int(len(dfk)*test_ratio),replace=False)
    test_set = set()
    for i in range(a.shape[0]):
        test_set.add(dfk.loc[a[i]][indexK])
    train_mask = torch.Tensor(size=[len(dfw)]).bool()
    test_mask = torch.Tensor(size=[len(dfw)]).bool()
    for i in range(len(dfw)):
        if dfw.loc[i][indexK] in test_set:
            test_mask[i] = True
            train_mask[i] = False
        else:
            test_mask[i] = False
            train_mask[i] = True
    return train_mask,test_mask

data = pd.read_csv('./DataSet/book/silver/claims.txt',sep='\t')
silverLabel = pd.read_csv('./DataSet/book/goldenNsilver.txt',sep='\t')
train_mask,test_mask = divideDataSet(data,silverLabel,0.6)
torch.save(test_mask,'./Result/test_mask.txt')
torch.save(train_mask,'./Result/train_mask.txt')
