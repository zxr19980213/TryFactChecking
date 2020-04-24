# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:17:36 2020

@author: zxr
"""

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl import DGLGraph

class NodeApplyModule(nn.Module):
    def __init__(self,in_feats,out_feats,activation):
        super(NodeApplyModule,self).__init__()
        self.linear = nn.Linear(in_feats,out_feats)
        self.activation = activation
    def forward(self,node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h,inplace=True)
        return {'h':h}
    
class GCN(nn.Module):
    def __init__(self,in_feats,out_feats,activation):
        super(GCN,self).__init__()
        self.gcn_msg = fn.copy_src(src='h',out='m')
        self.gcn_reduce = fn.sum(msg='m',out='h')
        self.apply_mod = NodeApplyModule(in_feats,out_feats,activation)
    def forward(self,g,feature):
        g.ndata['h'] = feature
        g.update_all(self.gcn_msg,self.gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')
    
class Net(nn.Module):
    def __init__(self,in_feats,out_feats,heads=1):
        super(Net,self).__init__()
        self.heads = heads
        self.fc1 = []
        for i in range(heads):
            self.fc1.append(nn.Linear(in_feats//heads,96))
        self.gcn2 = GCN(96*heads,96,F.relu)
        self.fc3 = nn.Linear(96,out_feats)
        #self.fc3 = nn.Linear(96*self.heads,out_feats)
    def forward(self,g,features):
        x=[]
        length = features.shape[1]//self.heads
        for i in range(self.heads):
            x.append( self.fc1[i](features[:,i*length:(i+1)*length]) )
        x = torch.cat(x,-1)
        x = self.gcn2(g,x)
        x = self.fc3(x)
        return x
    def predict(self,pred_prob):
        pred = F.softmax(pred_prob)
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)