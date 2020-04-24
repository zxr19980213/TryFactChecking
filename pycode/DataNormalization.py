# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:43:30 2020

@author: zxr
"""

import pandas as pd
pd.options.display.max_columns = 10
pd.options.display.max_rows = 20
pd.set_option('display.width', 1000)

def sim_jaccard (str1,str2) :
    set1 = set( str1.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    set2 = set( str2.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    return len(set1&set2)/len(set1|set2)

def data_normalization(df,key_col='isbn',answer_col='author'):
    df_new = df.copy()
    keys = df_new[key_col].unique()
    for key in keys:
        indices = df_new[key_col]==key
        df_temp = df_new[indices]
        answers = []
        for index,row in df_temp.iterrows():
            df_new.loc[index,answer_col] = df.loc[index,answer_col].lower()
            if answers.count(df.loc[index,answer_col]):
                continue
            found = False
            for i in range(len(answers)):
                if sim_jaccard(answers[i],df.loc[index,answer_col])>=0.80:
                    df_new.loc[index,answer_col] = answers[i]
                    found = True
                    break
            if not found:
                answers.append(df_new.loc[index,answer_col])
    return df_new
