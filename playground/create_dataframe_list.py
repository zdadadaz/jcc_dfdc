#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:57:16 2020

@author: liulara
"""

import os
import pandas as pd
from sklearn.utils import resample


def assign_split_by_index(df, real_re, split):
    index_list = real_re.index
    for i in range(len(index_list)):
        df.iloc[index_list[i],2] = split


def sample_data_by_folder(dfdc_foldername, split, df, seed):
#    dfdc_foldername = "dfdc_train_part_12"
#    split = "train"
    print(dfdc_foldername)
    real_num = sum((df['label']=="REAL") & (df['folder']==dfdc_foldername) & (df['frame_num']>=5))
    fake_num = sum((df['label']=="FAKE") & (df['folder']==dfdc_foldername) & (df['frame_num']>=5))
    if real_num ==0 and fake_num == 0:
        print("no valid data")
        return 
    real_num_extract = int(real_num * real_ratio)
    fake = df[(df['label']=="FAKE") & (df['folder']==dfdc_foldername) & (df['frame_num']>=5)]
    real = df[(df['label']=="REAL") & (df['folder']==dfdc_foldername) & (df['frame_num']>=5)]
    if real_ratio != 1:
        real_re = resample(real, n_samples=real_num_extract, replace=False, random_state=seed)
    else:
        real_re = real
    fake_re = resample(fake, n_samples=real_num_extract, replace=False, random_state=(seed+300))
    assign_split_by_index(df, real_re, split)
    assign_split_by_index(df, fake_re, split)

# maybe can be random
def create_folder_list(start, end):
    train_folder_list = []
    for i in range(start, end, 1):
        str_tmp = "dfdc_train_part_"
        train_folder_list.append(str_tmp+str(i))
    return train_folder_list

def loop_for_sample_data(train_folder_list, split, df):
#    dfdc_foldername = "dfdc_train_part_12"
#    split = "train"
    for i in range(len(train_folder_list)):
        seed = 500 + i
        dfdc_foldername = train_folder_list[i]
        sample_data_by_folder(dfdc_foldername, split, df, seed)

def check_frame_number(folder, df):
    dict_vid = {}
    for i in range(len(df)):
        dict_vid[df.iloc[i,0]]=i
    
    df['frame_num']=[0 for i in range(len(df))]
    split_folders = os.listdir(folder)
    for sf in split_folders:
        if sf[0] == ".":
            continue
        for df_real in os.listdir(os.path.join(folder, sf)):
            if df_real[0] == ".":
                continue
            for f in os.listdir(os.path.join(folder, sf, df_real)):
                if f[0] == ".":
                    continue
                vid = f.split('_')[0]
                df.iloc[dict_vid[vid], 5] += 1

def create_training_file(train_df):
    filenmes = []
    labels = []   
    for i in range(len(train_df)):
        vid = train_df.iloc[i,0]
        folder = train_df.iloc[i,4]
        label = train_df.iloc[i,1]
        fn = train_df.iloc[i,5]
        step = fn//5
        count = 0
        for s in range(0,fn,step):
            count += 1
            file = vid + "_"+ str(s)+".jpg"
            tmp_str = os.path.join(folder,label,file)
            filenmes.append(tmp_str)
            labels.append(label)
        if count != 5:
            print("error, frame number is less than 5")
    dict_assign = {'filenme': filenmes, 'label': labels}
    df = pd.DataFrame(dict_assign, columns = ['filename', 'label'])
    return df

folder = "./../db_multi_playground"
df_file= "./../metadata_small.csv"

train_folder_number = 35
valid_folder_number = 10
test_folder_number = 5

real_ratio = 1


df = pd.read_csv(df_file)                
df = df.drop(columns=['video.@width', 'video.@height'])
df.set_index('filename')

#initialize frame number
df['split']=["None" for i in range(len(df))]
#df['frame_num']=[15 for i in range(len(df))]
print("check frame number")
check_frame_number(folder, df)

print(" Sample dataframe")
train_folder_list = create_folder_list(0, train_folder_number)
valid_folder_list = create_folder_list(train_folder_number, train_folder_number + valid_folder_number)
test_folder_list = create_folder_list(train_folder_number + valid_folder_number, train_folder_number + valid_folder_number+test_folder_number)
print("sample for training")
loop_for_sample_data(train_folder_list, "train", df)
#print("sample for validation")
#loop_for_sample_data(valid_folder_list, "valid", df)
#print("sample for testing")
#loop_for_sample_data(test_folder_list, "test", df)

#df.sort_values('filename').to_csv('training_dataset.csv', index=False)

train_df = df[df['split']=='train']
print("create output file for training")
out_train_df = create_training_file(train_df)
#out_train_df.sort_values('filenme').to_csv('training_dataset.csv', index=False)








