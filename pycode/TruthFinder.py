# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:42:47 2020

@author: zxr
"""

import numpy as np
from numpy.linalg import norm
import math
import pandas as pd

def sigmoid(x):
    return  1 / ( 1 + math.exp(-x) )

#imp(str1->str2)
def imp_Jaccard (str1,str2) :
    set1 = set( str1.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    set2 = set( str2.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    imp_rate = len(set1&set2)/len(set1)
    return imp_rate-0.5

class TruthFinder(object):    
    def __init__(self,implication,dampening_factor=0.3,influence_related=0.5,source_col='source',key_col='isbn',ans_col='author'):
        assert(0 < dampening_factor < 1)
        assert(0 <= influence_related <= 1)
        self.implication = implication
        self.dampening_factor = dampening_factor
        self.influence_related = influence_related
        self.source_col = source_col
        self.key_col = key_col
        self.ans_col = ans_col
        
    def train(self,dataframe,max_iterations=10,
              threshold=1e-4,initial_trustworthiness=0.5):
        dataframe["trustworthiness"]=\
            np.ones(len(dataframe.index))*initial_trustworthiness
        dataframe["fact_confidence"] = np.zeros(len(dataframe.index))
        for i in range(max_iterations):
            print('iteration',i)
            t1 = dataframe.drop_duplicates( self.source_col )["trustworthiness"]
            dataframe = self.iteration(dataframe)
            t2 = dataframe.drop_duplicates( self.source_col )["trustworthiness"]
            if self.stop_condition(t1,t2,threshold*len(dataframe)):
                return dataframe
        return dataframe
        
    def iteration(self,df):
        df = self.update_fact_confidence(df)
        df = self.update_website_trustworthiness(df)
        return df
    
    def stop_condition(self,t1,t2,threshold):
        return norm(t2-t1)<threshold
        
    def update_fact_confidence(self,df):
        for object_ in df[ self.key_col ].unique():
            indices = df[ self.key_col ] == object_
            d = df.loc[indices]
            d = self.calculate_confidence(d)
            d = self.adjust_confidence(d)
            df.loc[indices] = self.compute_fact_confidence(d)
        return df
    
    def calculate_confidence(self,df):
        #Eq 3,5
        truthworthiness_score = lambda x: -math.log(1.0-x+1e-3)
        for i,row in df.iterrows():
            ts = df.loc[df[ self.ans_col ]==row[ self.ans_col ],"trustworthiness"]
            v = sum(truthworthiness_score(t) for t in ts)
            df.loc[i,'fact_confidence'] = v
        return df
    
    def adjust_confidence(self,df):
        #Eq 6
        update = {}
        for i,row1 in df.iterrows():
            f1 = row1[ self.ans_col ]
            s = 0
            for j,row2 in df.drop_duplicates( self.ans_col ).iterrows():
                f2 = row2[ self.ans_col ]
                if f1==f2:
                    continue
                s += row2["fact_confidence"] * self.implication(f2,f1)
            update[i] = self.influence_related * s + row1["fact_confidence"]
        for i,row1 in df.iterrows():
            df.loc[i,'fact_confidence'] = update[i]
        return df
    
    def compute_fact_confidence(self,df):
        #Eq 8
        f = lambda x: sigmoid(self.dampening_factor*x)
        for i,row in df.iterrows():
            df.loc[i,'fact_confidence'] = f(row['fact_confidence'])
        return df
    
    def update_website_trustworthiness(self,df):
        #Eq 1
        for website in df[ self.source_col ].unique():
            indices = df[ self.source_col ]==website
            cs = df.loc[indices,"fact_confidence"]
            df.loc[indices,"trustworthiness"] = sum(cs)/len(cs)
        return df
    
    def predict(self,df):
        df_ans = pd.DataFrame(columns=[self.key_col,self.ans_col])
        keys = df[self.key_col].unique()
        for key in keys:
            indices = df[self.key_col]==key
            df_temp = df[indices]
            max_fact = df_temp.iloc[0]['fact_confidence']
            max_ans = df_temp.iloc[0][self.ans_col]
            for index,row in df_temp.iterrows():
                if row['fact_confidence']>max_fact:
                    max_fact = row['fact_confidence']
                    max_ans = row[self.ans_col]
            df_ans = df_ans.append({self.key_col:key,self.ans_col:max_ans},ignore_index=True)
        return df_ans


df = pd.read_csv( './DataSet/book/silver/claims_normalization.txt' , sep='\t' )
finder = TruthFinder(imp_Jaccard,dampening_factor=0.3,influence_related=0.5)
df = finder.train(df)
df_ans = finder.predict(df)
df_ans.to_csv( './Result/TruthFinder_result.txt' , sep='\t' , index=False )