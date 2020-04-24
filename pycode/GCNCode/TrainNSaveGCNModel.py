# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:46:56 2020

@author: zxr
"""

import DefineModule as myModel
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import pickle
from sklearn.metrics import accuracy_score


whole_feature = torch.load('./DataSet/book/silver/feature.pt')
whole_label = torch.load('./DataSet/book/silver/label.pt')
whole_graph = pickle.load(open('./DataSet/book/silver/graph.pickle','rb'))

train_mask = torch.load('./Result/train_mask.txt')
test_mask = torch.load('./Result/test_mask.txt')


net = myModel.Net(whole_feature.shape[1],2,heads=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


for epoch in range(100):
    
    pred_prob = net.forward(whole_graph, whole_feature)
    loss = criterion(pred_prob[train_mask],whole_label[train_mask])
    
    #pred_prob = net.forward(graph_train, train_feature)
    #loss = criterion(pred_prob,train_label)
    #pred_label = net.predict(graph_whole, whole_feature)
    pred_label = net.predict(pred_prob)
    train_accu = accuracy_score(pred_label[train_mask],whole_label[train_mask])
    test_accu = accuracy_score(pred_label[test_mask],whole_label[test_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Epoch {:05d} | Loss {:.4f} | Train_Accu {:4f} | Test_Accu {:4f}".format(
        epoch, loss.item(), train_accu,test_accu))
torch.save(pred_prob,'./Result/gnn_prob.txt')
torch.save(net,'./Result/model.pt')