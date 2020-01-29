#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:31:57 2020

@author: liulara
"""

import pandas as pd
import numpy as np
import os 

dir_path = "./../xception_result"

filenames=[]
folders =[]
labels = []
splits = []
paths = []
for folder in os.listdir(dir_path):
    for df_real in os.listdir(os.path.join(dir_path,folder)):
        for file in os.listdir(os.path.join(dir_path,folder,df_real)):
#            print(file)
            filenames.append(file)
            folders.append(folder)
            labels.append(df_real)
            splits.append("train")
            paths.append(os.path.join(folder,df_real,file))
data = {"filename":filenames , "folder":folders, "label":labels, "split":splits, "path":paths}
df = pd.DataFrame(data,columns = ['filename','folder','label', 'split', 'path'])
df.sort_values('filename').to_csv('xception_list.csv', index=False)
        