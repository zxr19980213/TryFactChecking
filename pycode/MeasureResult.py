# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 08:51:50 2020

@author: zxr
"""

import pandas as pd

def sim_jaccard (str1,str2) :
    set1 = set( str1.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    set2 = set( str2.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    return len(set1&set2)/len(set1|set2)

def measure_result(df_predict,df_answer,answer_col='author'):
    ans_dict = {'0.0-0.2':0,'0.2-0.4':0,'0.4-0.6':0,'0.6-0.8':0,'0.8-1.0':0}
    miss_count = 0
    measure_sum = 0
    measure_hit = 0
    for index,row in df_predict.iterrows():
        if index not in df_answer.index:
            miss_count = miss_count + 1
        else:
            str1 = row[answer_col]
            str2 = df_answer.loc[index,answer_col]
            simmality = sim_jaccard(str1,str2)
            if simmality>=0.8:
                measure_hit+=1
                ans_dict['0.8-1.0']+=1
            elif simmality>=0.6:
                ans_dict['0.6-0.8']+=1
            elif simmality>=0.4:
                ans_dict['0.4-0.6']+=1
            elif simmality>=0.2:
                ans_dict['0.2-0.4']+=1
            else:
                ans_dict['0.0-0.2']+=1
            measure_sum += simmality
    print('miss_count',miss_count)
    print('measure_sum',measure_sum)
    print('measure_hit',measure_hit)
    print('ans_dict',str(ans_dict))

file_name = 'Result/MajorityVoting_result.txt'
print(file_name)
df_predict = pd.read_csv(file_name,sep='\t',index_col='isbn')
df_answer = pd.read_csv('DataSet/book/goldenNsilver.txt',sep='\t',index_col='isbn')
measure_result(df_predict,df_answer)