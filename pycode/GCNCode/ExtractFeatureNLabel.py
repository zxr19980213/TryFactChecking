# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:51:27 2020

@author: zxr
"""

import torch
import pandas as pd

def extractFeatureNLabel(df,indexE='encode',indexL='label'):
    a = torch.zeros(1,len(df.loc[0][indexE]))
    c = torch.zeros(1)
    for _,row in df.iterrows():
        b = row[indexE].reshape([1,-1])
        a = torch.cat((a,b),0)
        if row[indexL]:
            d = torch.ones(1)
        else:
            d = torch.zeros(1)
        c = torch.cat((c,d),-1)
    return a[1:,:],c[1:].long()

data = pd.read_csv('./DataSet/book/silver/claims_normalization.txt',sep='\t')
data['encode'] = torch.load('./DataSet/book/silver/bertEncode.pt')
feature,label = extractFeatureNLabel(data)
torch.save(feature,'./DataSet/book/silver/feature.pt')
torch.save(label,'./DataSet/book/silver/label.pt')