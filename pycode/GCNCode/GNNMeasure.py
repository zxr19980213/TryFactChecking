# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:18:46 2020

@author: zxr
"""

import pandas as pd
import torch
import torch.nn.functional as F

def add_confidence(df,prob,col_name='fact_confidence'):
    df[col_name] = None
    for i in range(len(df)):
        df.loc[i,col_name] = float(prob[i][1])
    return df

def sim_Jaccard (str1,str2) :
    set1 = set( str1.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    set2 = set( str2.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    return len(set1&set2)/len(set1|set2)

def MV(df,indexK='isbn',answer='author',withWeight=False,weight='confidence'):
    df_mv = pd.DataFrame(columns=[indexK,answer])
    for indexV in df[indexK].unique():
        data_slice = df[df[indexK]==indexV]
        vote_dict = {}
        for index,row in data_slice.iterrows():
            flag = False
            for key in vote_dict.keys():
                if ( sim_Jaccard(key,row[answer])>=0.8 ):
                    flag = True
                    if(not withWeight):
                        vote_dict[key] += 1
                    else:
                        vote_dict[key] += float(row[weight])
                    break
            if (not flag):
                if(not withWeight):
                    vote_dict[row[answer]] = 1
                else:
                    vote_dict[row[answer]] = float(row[weight])
        vote_list = sorted(vote_dict.items(), key=lambda d:d[1],reverse=True)
        #print({indexK:indexV,answer:vote_list[0][0]})
        df_mv = df_mv.append({indexK:indexV,answer:vote_list[0][0]},ignore_index=True)
    return df_mv

def JudgeAccu(label,pred,pred_col='author'):
    score1 = 0
    score2 = 0
    for index,row in pred.iterrows():
        if not(index in label.index):
            print(index,'no answer')
            score1 += 0 
            score1 += 0
        elif sim_Jaccard(row[pred_col],label.loc[index][pred_col])>=0.8:
            score1 +=1
            score2 +=1
        else:
            print(row[pred_col],"vs",label.loc[index][pred_col],sim_Jaccard(row[pred_col],label.loc[index][pred_col]))
            score1 += 0
            score2 += sim_Jaccard(row[pred_col],label.loc[index][pred_col])
    return score1/len(pred),score2/len(pred)

data = pd.read_csv('./DataSet/book/silver/claims.txt',sep='\t')
pred_prob = torch.load('./Result/gnn_prob.txt')
test_mask = torch.load('./Result/test_mask.txt')

data_withConfidence = add_confidence(data,F.softmax(pred_prob))

df_mv = MV(data_withConfidence[test_mask.numpy()],withWeight=True,weight='fact_confidence')
df_mv.to_csv( './Result/gnn_result.txt' , sep='\t' , index=False )

label = pd.read_csv('./DataSet/book/goldenNsilver.txt',sep='\t',low_memory=False,index_col=0)
pred = pd.read_csv('./Result/gnn_result.txt',sep='\t',low_memory=False,index_col=0)

print(JudgeAccu(label,pred))