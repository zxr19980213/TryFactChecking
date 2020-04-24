# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:47:01 2020

@author: zxr
"""

# train_mask = ~(test_mask)

import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import DefineGCNModule as myModel
from sklearn.metrics import accuracy_score

def k_fold(df_answer,key_col='isbn',k=5):
    divided_answer = []
    df_answer = df_answer.reset_index()
    for i in range(k):
        indices = df_answer.index%k==i
        divided_answer.append(df_answer[indices].set_index(key_col,drop=True))
    return divided_answer

def mask_dataset(df,df_mask,key_col='isbn'):
    mask_keys = set()
    for index,row in df_mask.iterrows():
        mask_keys.add(index)
    mask = torch.Tensor(size=[len(df)]).bool()
    for i in range(len(df)):
        if df.iloc[i][key_col] in mask_keys:
            mask[i] = True
        else:
            mask[i] = False
    return mask

def add_confidence(df,prob,col_name='fact_confidence'):
    prob = F.softmax(prob)
    df[col_name] = None
    for i in range(len(df)):
        df.loc[i,col_name] = float(prob[i][1])
    return df

def majority_voting(df,key_col='isbn',answer_col='author',with_weight=True,weight_col='fact_confidence'):
    df_mv = pd.DataFrame(columns=[key_col,answer_col])
    for key in df[key_col].unique():
        indices = df[key_col]==key
        max_vote = -1
        max_answer = df[indices].iloc[0][answer_col]
        for answer in df[indices][answer_col].unique():
            indices2 = indices&(df[answer_col]==answer)
            vote = 0
            if(not with_weight):
                vote = sum(indices2)
            else:
                vote = sum(df[indices2][weight_col])
                #vote = sum(df[indices2][weight_col],sum(indices2)/sum(indices))
            if vote>=max_vote:
                #print('reach here')
                max_vote = vote
                max_answer = answer
        df_mv = df_mv.append({key_col:key,answer_col:max_answer},ignore_index=True)
    return df_mv

whole_feature = torch.load('../DataSet/book/silver/feature.pt')
whole_label = torch.load('../DataSet/book/silver/label.pt')
whole_graph = pickle.load(open('../DataSet/book/silver/graph.pickle','rb'))

#layers-3 train_rate-0.2 epoch-150 heads-3 rate-1e-4 82.8
#-------- -----------0.8 --------------------------- 

kfolds=8
path = '../Result/'+str(kfolds)+'folds/'
if not os.path.exists(path):
    os.makedirs(path)
    print('create path',path)
data = pd.read_csv('../DataSet/book/silver/claims_normalization.txt',sep='\t')
answer = pd.read_csv('../DataSet/book/goldenNsilver.txt',sep='\t',index_col='isbn')
divided_answer = k_fold(answer,k=kfolds)
for i in range(kfolds):
    print('fold',i,'in',kfolds,'folds start')
    test_mask = mask_dataset(data,divided_answer[i])
    train_mask = ~(test_mask)
    net = myModel.Net(whole_feature.shape[1],2,heads=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-4)
    for epoch in range(200):
        pred_prob = net.forward(whole_graph,whole_feature)
        loss = criterion(pred_prob[train_mask],whole_label[train_mask])
        
        pred_label = net.predict(pred_prob)
        train_accu = accuracy_score(pred_label[train_mask],whole_label[train_mask])
        test_accu = accuracy_score(pred_label[test_mask],whole_label[test_mask])
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        print("Epoch {:05d} | Loss {:.4f} | Train_Accu {:4f} | Test_Accu {:4f}".format(
                epoch, loss.item(), train_accu,test_accu))
    data = add_confidence(data,pred_prob)
    test_answer = majority_voting(data[test_mask.numpy()])
    file_name = path + 'answer'+ str(i) + '.txt'
    test_answer.to_csv(file_name,sep='\t',index=False)
    print('fold',i,'answer saved')