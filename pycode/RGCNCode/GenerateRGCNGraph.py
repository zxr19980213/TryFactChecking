# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:40:19 2020

@author: zxr
"""

import pandas as pd
from dgl import DGLGraph
import pickle

def sim_Jaccard (str1,str2) :
    set1 = set( str1.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    set2 = set( str2.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    return len(set1&set2)/len(set1|set2)

def generateDGLGraph(df,index0='isbn',index1='source',index2='author'):
    g = DGLGraph()
    g.add_nodes(len(df))
    
    index0_list = df[index0].drop_duplicates().reset_index(drop=True)
    index1_list = df[index1].drop_duplicates().reset_index(drop=True)
    
    for index,_ in df.iterrows():
        g.add_edge(index,index)
    
    dict1 = {}
    dict2 = {}
    
    print('total',index1,'number:',len(index1_list))
    for index,value in index1_list.iteritems():
        df_slice = df[df[index1]==value]
        dict1[value] = len(df_slice)
        print(index1,value,'with',len(df_slice),'claims')
        for i in range(len(df_slice)):
            for j in range(i+1,len(df_slice)):
                g.add_edge(df_slice.iloc[i].name,df_slice.iloc[j].name)
                g.add_edge(df_slice.iloc[j].name,df_slice.iloc[i].name)
       
    print('total',index0,'number:',len(index0_list))         
    for index,value in index0_list.iteritems():
        df_slice = df[df[index0]==value]
        dict2[value] = len(df_slice)
        print(index0,value,'with',len(df_slice),'claims')
        for i in range(len(df_slice)):
            for j in range(i+1,len(df_slice)):
                if sim_Jaccard(df_slice.iloc[i][index2],df_slice.iloc[j][index2])>=0.8:
                    g.add_edge(df_slice.iloc[i].name,df_slice.iloc[j].name)
                    g.add_edge(df_slice.iloc[j].name,df_slice.iloc[i].name)
    return g,dict1,dict2

data = pd.read_csv('./DataSet/book/silver/claims.txt',sep='\t')
g,dict1,dict2 = generateDGLGraph(data)
print('total nodes:',g.number_of_nodes())
print('total edges:',g.number_of_edges())

file = open('./DataSet/book/silver/graph.pickle','wb')
pickle.dump(g,file)
file.close()

file = open('./DataSet/book/silver/sourceDistribution.txt','wb')
pickle.dump(dict1,file)
file.close()

file = open('./DataSet/book/silver/isbnDistribution.txt','wb')
pickle.dump(dict2,file)
file.close()