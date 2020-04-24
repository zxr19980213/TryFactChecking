# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:05:09 2020

@author: zxr
"""

# df = pd.read_csv("claims_golden.txt",sep='\t')

import numpy as np
from numpy.linalg import norm
import pandas as pd

pd.options.display.max_columns = 10
pd.options.display.max_rows = 20


class PooledInvestment(object):
    def __init__(self,g):
        self.g = g
        assert( 1 <= self.g <= 2 )

    def train(self,df,source_col='source',key_col='isbn',
              answer_col='author',init_trust=0.5,max_iteration=10,threshold=1e-1):
        self.source_col = source_col
        self.key_col = key_col
        self.answer_col = answer_col
        
        df['trustworthiness'] = np.ones(len(df))*init_trust
        df['fact_confidence'] = np.zeros(len(df))
        df = self.initFactConfidence(df)
        print('sum of claim confidence :',sum(df['fact_confidence']))
        
        for i in range(max_iteration):
            print('iteration',i)
            self.old_df = df.copy()
            t1 = df.drop_duplicates(self.source_col)['trustworthiness']
            df = self.iteration(df)
            print('sum of claim confidence :',sum(df['fact_confidence']))
            t2 = df.drop_duplicates(self.source_col)['trustworthiness']
            if self.stop_condition(t1,t2,threshold):
                return df              
        return df
    
    def iteration(self,df):
        df = self.computeSourceTrust(df)
        df = self.computeClaimConfidence(df)
        return df
    
    def stop_condition(self,t1,t2,threshold):
        delta = norm(t2-t1,ord=1)
        sum_ori = norm(t1,ord=1)
        print('delta',delta,delta/sum_ori)
        return delta<=threshold*sum_ori
    
    def computeClaimConfidence(self,df):
        keys = df[self.key_col].unique()
        print('sum of different keys :',len(keys))
        for key in keys:
            sum_temp = 0
            indices = df[self.key_col]==key
            df_temp = df[indices]
            answers = df_temp[self.answer_col].unique()
            for answer in answers:
                hc = 0
                indices2 = indices&(df[self.answer_col]==answer)
                #indices2 = df_temp[self.answer_col]==answer
                df_temp2 = df[indices2]
                for index2,row2 in df_temp2.iterrows():
                    temp_size = sum(df[self.source_col]==row2[self.source_col])
                    hc = hc + row2['trustworthiness']/temp_size
                df.loc[indices2] = self.setConfidence(df.loc[indices2],hc)
                #df_temp.loc[indices2] = self.setConfidence(df_temp2,hc)
                sum_temp = sum_temp + pow(hc,self.g)
            for answer in answers:
                indices2 = indices&(df[self.answer_col]==answer)
                #indices2 = df_temp[self.answer_col]==answer
                answer_confidence = df[indices2].iloc[0]['fact_confidence']
                answer_confidence = answer_confidence*pow(answer_confidence,self.g)/sum_temp
                df.loc[indices2] = self.setConfidence(df.loc[indices2],answer_confidence)
        return df
                
    
    def computeSourceTrust(self,df):
        sources = df[self.source_col].unique()
        print('sum of different sources :',len(sources))
        for source in sources:
            t_temp = 0
            indices = df[self.source_col]==source
            indices1 = self.old_df[self.source_col]==source
            df_temp = self.old_df[indices1]
            claimSize = sum(indices1)
            for index,row in df_temp.iterrows():
                claimIndex = row[self.key_col]
                claimValue = row[self.answer_col]
                s_temp = 0
                indices2 = (self.old_df[self.key_col]==claimIndex)&(self.old_df[self.answer_col]==claimValue)
                df_temp2 = self.old_df[indices2]
                for index2,row2 in df_temp2.iterrows():
                    temp_size = sum(self.old_df[self.source_col]==row2[self.source_col])
                    s_temp = s_temp + row2['trustworthiness']/temp_size
                if(s_temp==0.0):
                    t_temp=0
                    break
                t_temp = t_temp + row['fact_confidence']*row['trustworthiness']/(claimSize*s_temp)
            df.loc[indices] = self.setTrustWorthiness(df.loc[indices],t_temp)
        return df
    
    def setTrustWorthiness(self,df,t):
        for index,row in df.iterrows():
            df.loc[index,'trustworthiness']=t
        return df
    
    def setConfidence(self,df,c):
        for index,row in df.iterrows():
            df.loc[index,'fact_confidence']=c
        return df
    
    def initFactConfidence(self,df):
        for object_ in df[self.key_col].unique():
            indices = df[self.key_col]==object_
            values = df[indices][self.answer_col].unique()
            df.loc[indices] = self.setConfidence(df.loc[indices],1/len(values))
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
p = PooledInvestment(1.4)
df_t = p.train(df,max_iteration=10)
df_ans = p.predict(df)
df_ans.to_csv('./Result/PooledInvestment_result.txt',sep='\t',index=False)