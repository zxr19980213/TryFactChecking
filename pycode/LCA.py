# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:03:01 2020

@author: zxr
"""

import pandas as pd
import numpy as np
from numpy.linalg import norm

#give_up_sadly
#Latent Credibility Analysis
#这个方法整出的结果粗看有些问题？
#1-188;2-171;3-160
def sim_jaccard (str1,str2) :
    set1 = set( str1.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    set2 = set( str2.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    return len(set1&set2)/len(set1|set2)

def imp_Jaccard (str1,str2) :
    set1 = set( str1.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    set2 = set( str2.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    imp_rate = len(set1&set2)/len(set1)
    return imp_rate-0.5

class LCA(object):
    def __init__(self,beta=0.5,threshold=0.5):
        self.beta = beta
        self.threshold = threshold
    
    def train(self,df,source_col='source',
              key_col='isbn',answer_col='author',
              initial_trustworthiness=0.5,initial_confidence=0.5,
              max_iteration=10,threshold=1e-2):
        self.source_col = source_col
        self.key_col = key_col
        self.answer_col = answer_col
        df["trustworthiness"]=np.ones(len(df))*initial_trustworthiness
        df['fact_confidence']=np.ones(len(df))*initial_confidence
        for i in range(max_iteration):
            print('iteration',i)
            t1 = df.drop_duplicates(self.source_col)['trustworthiness']
            df = self.iteration(df)
            t2 = df.drop_duplicates(self.source_col)['trustworthiness']
            if self.stop_condition(t1,t2,threshold):
                return df
        return df
    
    def stop_condition(self,t1,t2,threshold):
        delta = norm(t2-t1,ord=1)
        sum_ori = norm(t1,ord=1)
        print('delta',delta,delta/sum_ori)
        return delta<=threshold*sum_ori
    
    def iteration(self,df):
        df = self.compute_confidence(df)
        df = self.compute_trustworthiness(df)
        return df
    
    
    '''
compute_sourcetrust:
    for each source s:
        t=0
        for each claim c of s:
            t += c.confidence
        t /= |s.claims|
        s.trust = t
    '''
    def compute_trustworthiness(self,df):
        sources = df[self.source_col].unique()
        for source in sources:
            indices = df[self.source_col]==source
            trust = 0.0
            for index,row in df[indices].iterrows():
                trust = trust + row['fact_confidence']
            trust = trust/sum(indices)
            trust = min(trust,1)
            df.loc[indices,'trustworthiness']=trust
        return df
    
    '''
compute_factconfidence:
    for each object o:
        m = |o.values|
        for each value v of o:
            f = 1
            for each source s of v:
                f *= s.trust
            for each exclusive value v_ of v:
                for each source s of v_:
                    f *= (1-s.trust)/(m-1)
            v.confidence
    '''
    def compute_confidence(self,df):
        keys = df[self.key_col].unique()
        for key in keys:
            indices = df[self.key_col]==key
            answers = df[indices][self.answer_col].unique()
            m = len(answers)
            if m==1:
                df.loc[indices,'fact_confidence'] = 1.0
            else:
                for answer in answers:
                    belief = 1.0
                    indices2 = indices&(df[self.answer_col]==answer)
                    for index,row in df[indices2].iterrows():
                        belief = belief * (row['trustworthiness'])
                        #belief = belief * (row['trustworthiness']+1e-2)
                    indices3 = indices&(df[self.answer_col]!=answer)
                    for index,row in df[indices3].iterrows():
                        exclusive_answer = row[self.answer_col]
                        #belief *= sim_jaccard(answer,exclusive_answer)*row['trustworthiness']
                        if sim_jaccard(answer,exclusive_answer)<=self.threshold:
                            belief = belief * (1-row['trustworthiness'])*sim_jaccard(answer,exclusive_answer)
                            #belief = belief * (1-row['trustworthiness'])/(m-1)
                            #belief = belief * (1-row['trustworthiness']+1e-2)/(m-1)
                        else:
                            #belief = belief * row['trustworthiness'] * sim_jaccard(answer,exclusive_answer)
                            belief = belief * row['trustworthiness'] 
                    #df.loc[indices2,'fact_confidence'] = self.beta * belief
                    df.loc[indices2,'fact_confidence'] = self.beta * belief+ (1-self.beta)*df[indices2].iloc[0]['fact_confidence']
        return df
    
    def predict(self,df):
        df_ans = pd.DataFrame(columns=[self.key_col,self.answer_col])
        keys = df[self.key_col].unique()
        for key in keys:
            indices = df[self.key_col]==key
            df_temp = df[indices]
            max_fact = df_temp.iloc[0]['fact_confidence']
            max_ans = df_temp.iloc[0][self.answer_col]
            for index,row in df_temp.iterrows():
                if row['fact_confidence']>max_fact:
                    max_fact = row['fact_confidence']
                    max_ans = row[self.answer_col]
            df_ans = df_ans.append({self.key_col:key,self.answer_col:max_ans},ignore_index=True)
        return df_ans
    
df = pd.read_csv("DataSet/book/silver/claims_normalization.txt",sep='\t')
l = LCA(beta=0.8)
df_t = l.train(df)
df_ans = l.predict(df)
df_ans.to_csv('./Result/LCA_result.txt',sep='\t',index=False)