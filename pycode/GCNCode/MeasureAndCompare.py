# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:09:34 2020

@author: zxr
"""

import os
import numpy as np
import pandas as pd

def sim_jaccard (str1,str2) :
    set1 = set( str1.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    set2 = set( str2.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    return len(set1&set2)/len(set1|set2)

standard_answers = pd.read_csv('../DataSet/book/goldenNsilver.txt',sep='\t',index_col='isbn')
compare_method_counts = 3
mv_answers = pd.read_csv('../Result/MajorityVoting_result.txt',sep='\t',index_col='isbn')
pi_answers = pd.read_csv('../Result/PooledInvestment_result.txt',sep='\t',index_col='isbn')
tf_answers = pd.read_csv('../Result/TruthFinder_result.txt',sep='\t',index_col='isbn')
compare_method_names = ['MajorityVoting','PooledInvestment','TruthFinder']
compare_method_answers = [mv_answers,pi_answers,tf_answers]
gnn_hits = 0
gnn_sum = 0
compare_method_hits = [0,0,0]
compare_method_sum = [0,0,0]

count = 0
path = '../Result/2folds'
files = os.listdir(path)
for file in files:
    file_name = path+'/'+file
    gnn_answers = pd.read_csv(file_name,sep='\t',index_col='isbn')
    for index,row in gnn_answers.iterrows():
        index = str(index)
        count += 1
        correct_answer = standard_answers.loc[index]['author']
        sim_score = sim_jaccard(row['author'],correct_answer)
        gnn_sum += sim_score
        if sim_score>=0.8:
            gnn_hits+=1
        for i in range(compare_method_counts):
            compare_answer = compare_method_answers[i].loc[index]['author']
            sim_score = sim_jaccard(compare_answer,correct_answer)
            compare_method_sum[i] += sim_score
            if sim_score >= 0.8:
                compare_method_hits[i] += 1

print('total count',count)
print('gnn','hits:',gnn_hits,'sum',gnn_sum)
for i in range(compare_method_counts):
    print(compare_method_names[i],'hits',compare_method_hits[i],'sum',compare_method_sum[i])