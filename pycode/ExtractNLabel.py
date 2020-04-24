# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:42:13 2020

@author: zxr
"""

import pandas as pd

data = pd.read_table("./DataSet/book/book.txt" , sep='\t' , header=None , names=['source','isbn','name','author'])
data = data.drop_duplicates().reset_index(drop=True)
data.fillna('Not Available', inplace=True)
GoldenLabel = pd.read_table("./DataSet/book/book_golden.txt" , sep='\t' , header=None , names=['isbn','author'])
SilverLabel = pd.read_table("./DataSet/book/book_silver.txt" , sep='\t' , header=None , names=['isbn','author'])
SilverLabel = SilverLabel[SilverLabel['isbn']!='F'].reset_index(drop=True)

for index,row in data.iterrows():
    if len(row['isbn'])==10:
        row['isbn'] = '978'+row['isbn']
for index,row in GoldenLabel.iterrows():
    if len(row['isbn'])==10:
        row['isbn'] = '978'+row['isbn']

my_golden = pd.DataFrame(GoldenLabel)
golden_isbn = GoldenLabel['isbn'].unique()
silver_isbn = SilverLabel['isbn'].unique()
data_isbn = data['isbn'].unique()
count = 0
for index,row in SilverLabel.iterrows():
    if (row['isbn'] in data_isbn) and (row['isbn'] not in golden_isbn):
        my_golden = my_golden.append(row)
my_golden.reset_index(drop=True,inplace=True)
my_golden.to_csv('./DataSet/book/goldenNsilver.txt',sep='\t',index=False)

def sim_Jaccard (str1,str2) :
    set1 = set( str1.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    set2 = set( str2.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split() )
    return len(set1&set2)/len(set1|set2)

def labelFunc(data,answer,indexK='isbn',indexV='author'):
    data_golden = pd.DataFrame()
    for index1,row1 in answer.iterrows():
        str1 = row1[indexV]
        data_slice = pd.DataFrame(data[data[indexK]==row1[indexK]])
        data_slice['label']=False
        for index2,row2 in data_slice.iterrows():
            str2 = row2[indexV]
            data_slice.loc[index2,'label'] = ( sim_Jaccard(str1,str2) >= 0.8 )
        data_golden = data_golden.append(data_slice)
    return data_golden.reset_index(drop=True)
data_silver = labelFunc(data,my_golden)
data_silver.to_csv('./DataSet/book/silver/claims.txt',sep='\t',index=False)