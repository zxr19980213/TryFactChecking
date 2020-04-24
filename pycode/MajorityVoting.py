# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:39:25 2020

@author: zxr
"""

import pandas as pd

def sim_Jaccard (str1,str2) :
    set1 = set( str1.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    set2 = set( str2.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    return len(set1&set2)/len(set1|set2)

def MV(df,key_col='isbn',answer_col='author',withWeight=False,weight='fact_confidence'):
    df_mv = pd.DataFrame(columns=[key_col,answer_col])
    keys = df[key_col].unique()
    for key in keys:
        indices = df[key_col]==key
        answers = df[indices][answer_col].unique()
        max_answer = df[indices].iloc[0][answer_col]
        max_count = 1
        for answer in answers:
            indices2 = (df[key_col]==key)&(df[answer_col]==answer)
            count = sum(indices2)
            if count>max_count:
                max_count = count
                max_answer = answer
        df_mv = df_mv.append({key_col:key,answer_col:max_answer},ignore_index=True)
    return df_mv

df = pd.read_csv( './DataSet/book/silver/claims_normalization.txt' , sep='\t' )
df_ans = MV(df)
df_ans.to_csv( './Result/MajorityVoting_result.txt' , sep='\t' , index=False )