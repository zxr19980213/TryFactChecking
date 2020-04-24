# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:32:46 2020

@author: zxr
"""

import pandas as pd
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer,BertModel
import logging

logging.basicConfig(level=logging.INFO)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def feedColumnIntoBert(feature):
    for i in range(len(feature)):
        feature.loc[i] = '[CLS] '+feature.loc[i]+' [SEP]'
        feature.loc[i] = tokenizer.tokenize(feature.loc[i])
        feature.loc[i] = tokenizer.convert_tokens_to_ids(feature.loc[i])
        feature.loc[i] = torch.tensor([feature.loc[i]])
        with torch.no_grad():
            encoded_layers,_ = model(feature.loc[i])
        feature.loc[i] = encoded_layers[-1][0][0]+encoded_layers[-2][0][0]+encoded_layers[-3][0][0]+encoded_layers[-4][0][0]
    return feature

data = pd.read_csv('./DataSet/book/silver/claims_normalization.txt',sep='\t')
#data = pd.read_csv('./DataSet/book/silver/claims.txt',sep='\t')
feature1 = feedColumnIntoBert(pd.Series(data['source']))
feature2 = feedColumnIntoBert(pd.Series(data['name']))
feature3 = feedColumnIntoBert(pd.Series(data['author']))
for i in range(len(feature1)):
    feature1.loc[i] = torch.cat((feature1.loc[i],feature2.loc[i],feature3.loc[i]))
torch.save(feature1,'./DataSet/book/silver/bertEncode.pt')