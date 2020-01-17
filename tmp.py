#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 19:28:12 2020

@author: zdadadaz
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# dir_json = './../fb_whole/metadata_21.json'
# train_list =[]
# with open(dir_json) as json_file:
#     data = json.load(json_file)
#     train_list = pd.DataFrame.from_dict(data, orient='index')
#     train_list.reset_index(level=0, inplace=True)
    
# train_list[train_list['label']=='REAL'].iloc[1]

base = pd.read_csv('submission_base.csv')
mtcn = pd.read_csv('submission_mtcn.csv')
whole = pd.read_csv('metadata_small.csv')

sLength = len(base['label'])
base['wrong'] = pd.Series(np.random.randn(sLength), index=base.index)
base['original'] = pd.Series(np.random.randn(sLength), index=base.index)
base['folder'] = pd.Series(np.random.randn(sLength), index=base.index)
base['res'] = pd.Series(np.random.randn(sLength), index=base.index)
mtcn['wrong'] = pd.Series(np.random.randn(sLength), index=base.index)
mtcn['original'] = pd.Series(np.random.randn(sLength), index=base.index)
mtcn['folder'] = pd.Series(np.random.randn(sLength), index=base.index)
mtcn['res'] = pd.Series(np.random.randn(sLength), index=base.index)

for i in range(len(base)):
    print(str(i))
    fn = base.iloc[i][0]
    label = whole[whole['filename']==fn]['label']
    score =0
    origin = "n"
    folder = whole[whole['filename']==fn]['folder']
    if list(label)[0] =="FAKE":
        score = 1
        origin = whole[whole['filename']==fn]['original']
    
    base['wrong'][i]= abs(score - base.iloc[i][1])>0.5
    base['original'][i]= list(origin)[0]
    base['folder'][i]= list(folder)[0]
    base['res'][i]= list(label)[0]
    
    mtcn['wrong'][i]= abs(score - mtcn.iloc[i][1])>0.5
    mtcn['original'][i]= list(origin)[0]
    mtcn['folder'][i]= list(folder)[0]
    mtcn['res'][i]= list(label)[0]
    
for i, d in base.groupby('res'):
    base['label'].plot(kind='hist', figsize=(15, 5), bins=20, alpha=0.8, title='base')
    plt.legend(['FAKE','REAL'])
plt.show()
for i, d in base.groupby('res'):
    mtcn['label'].plot(kind='hist', figsize=(15, 5), bins=20, title='MTCNN', alpha=0.8)
    plt.legend(['FAKE','REAL'])
plt.show()

TP = sum(np.array(base['label']>0.5) & np.array(base['res']=="FAKE"))
FP = sum(np.array(base['label']>0.5) & np.array(base['res']=="REAL"))
TN = sum(np.array(base['label']<=0.5) & np.array(base['res']=="FAKE"))
FN = sum(np.array(base['label']<=0.5) & np.array(base['res']=="REAL"))
precision = TP/len(base)*2
recall = TP/(TP+FP)
Fake_precision = TP/(TP+TN)
Real_precision = FN/(FP+FN)