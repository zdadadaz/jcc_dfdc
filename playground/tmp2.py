#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:31:57 2020

@author: liulara
"""

import pandas as pd
import numpy as np
import os 

# dir_path = "./../xception_result"

# filenames=[]
# folders =[]
# labels = []
# splits = []
# paths = []
# for folder in os.listdir(dir_path):
#     for df_real in os.listdir(os.path.join(dir_path,folder)):
#         for file in os.listdir(os.path.join(dir_path,folder,df_real)):
# #            print(file)
#             filenames.append(file)
#             folders.append(folder)
#             labels.append(df_real)
#             splits.append("train")
#             paths.append(os.path.join(folder,df_real,file))
# data = {"filename":filenames , "folder":folders, "label":labels, "split":splits, "path":paths}
# df = pd.DataFrame(data,columns = ['filename','folder','label', 'split', 'path'])
# df.sort_values('filename').to_csv('xception_list.csv', index=False)

df_audio = pd.read_csv('./../metadata_audio_altered.csv')
df_file= pd.read_csv("./dataset_orginal.csv")
dict_file = {}
for i in range(len(df_file)):
    dict_file[df_file.iloc[i,0]] = i

droplist = []
for i in range(len(df_audio)):
    droplist.append(dict_file[df_audio.iloc[i,0]])
df_file_m = df_file.drop(droplist)
df_file_m.to_csv('./dataset_orginal_no_audio.csv', index=False)
