#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:57:16 2020

@author: liulara
"""

import os
import pandas as pd
from sklearn.utils import resample
import cv2

number_of_image_per_video = 5

def assign_split_by_index(df, real_re, split):
    index_list = real_re.index
    for i in range(len(index_list)):
        df.iloc[index_list[i],2] = split


def sample_data_by_folder(dfdc_foldername, split, df, seed):
#    dfdc_foldername = "dfdc_train_part_12"
#    split = "train"
    print(dfdc_foldername)
    number_of_valid_image = number_of_image_per_video
    real_num = sum((df['label']=="REAL") & (df['folder']==dfdc_foldername) & (df['frame_num']>=number_of_valid_image))
    fake_num = sum((df['label']=="FAKE") & (df['folder']==dfdc_foldername) & (df['frame_num']>=number_of_valid_image))
    # print("real number no cut = " + str( sum((df['label']=="REAL") & (df['folder']==dfdc_foldername))))
    # print("real number = " + str(real_num))
    if real_num ==0 and fake_num == 0:
        print("no valid data")
        return 
    real_num_extract = int(real_num * real_ratio)
    fake = df[(df['label']=="FAKE") & (df['folder']==dfdc_foldername) & (df['frame_num']>=number_of_valid_image)]
    real = df[(df['label']=="REAL") & (df['folder']==dfdc_foldername) & (df['frame_num']>=number_of_valid_image)]
    if real_ratio != 1:
        real_re = resample(real, n_samples=real_num_extract, replace=False, random_state=seed)
    else:
        real_re = real
    fake_re = resample(fake, n_samples=real_num_extract, replace=False, random_state=(seed+300))
    assign_split_by_index(df, real_re, split)
    assign_split_by_index(df, fake_re, split)
    return real_num_extract*2

# maybe can be random
def create_folder_list(arr):
    train_folder_list = []
    for i in arr:
        str_tmp = "dfdc_train_part_"
        train_folder_list.append(str_tmp+str(i))
    return train_folder_list

def loop_for_sample_data(train_folder_list, split, df):
#    dfdc_foldername = "dfdc_train_part_12"
#    split = "train"
    total_sum = 0
    for i in range(len(train_folder_list)):
        seed = 500 + i
        dfdc_foldername = train_folder_list[i]
        total_sum += sample_data_by_folder(dfdc_foldername, split, df, seed)
    print("Total number of "+split + " is "+str(total_sum))
        
def read_image_check_size(path):
    img = cv2.imread(path)
    size = img.shape
    if size[0] < 60 and size[1] < 60:
        return True
    return False

def check_frame_number(folder, df):
    dict_vid = {}
    for i in range(len(df)):
        dict_vid[df.iloc[i,0]]=i
    
    df['frame_num']=[0 for i in range(len(df))]
    split_folders = os.listdir(folder)
    for sf in split_folders:
        if sf[0] == "." :
            continue
        print("folder name: " +sf)
        for df_real in os.listdir(os.path.join(folder, sf)):
            if df_real[0] == ".":
                continue
            for f in os.listdir(os.path.join(folder, sf, df_real)):
                if f[0] == ".":
                    continue
                # img_path = os.path.join(folder, sf, df_real,f)
                # if read_image_check_size(img_path):
                #     cmd2 = 'rm ' + img_path
                #     os.system(cmd2)
                #     continue
                vid = f.split('_')[0]
                df.iloc[dict_vid[vid], 5] += 1


def re_order_image(vid, path, fn):
    if fn == 0:
        return
    cmd = 'ls -1 ' + os.path.join(path, vid+"*")
    tmp= os.popen(cmd).read()
    files = tmp.split("\n")[:-1]
    arr =[]
    for f in range(len(files)):
        if files[f] =="":
            continue
        num = int(files[f].split("_")[-1][:-4])
        arr.append((num,files[f].split("/")[-1]))
    sorted_aa = sorted(arr)
    if len(files) == (sorted_aa[-1][0]+1):
        return
    for i in range(len(sorted_aa)):
        cmd2 = 'mv ' + os.path.join(path, sorted_aa[i][1]) + " " + os.path.join(path, vid+"_"+str(i)+".jpg")
        print(cmd2)
        os.system(cmd2)

# =============================================================================
# # Need test 1
# =============================================================================
def loop_for_reorder(df, dir_path):
    for i in range(len(df)):
        print(str(i) +" / "+ str(len(df)))
        vid = df.iloc[i,0]
        folder = df.iloc[i,4]
        label = df.iloc[i,1]
        fn = df.iloc[i,5]
        path = os.path.join(dir_path, folder, label)
        re_order_image(vid, path, fn)
        
    
def create_training_file(train_df):
    filenmes = []
    labels = []   
    number_of_image_per_video_local = number_of_image_per_video
    mapping2pervideo = [0]*16
    mapping2pervideo[15]=6
    mapping2pervideo[14]=6
    mapping2pervideo[13]=6
    mapping2pervideo[12]=6
    mapping2pervideo[11]=3
    mapping2pervideo[10]=3
    mapping2pervideo[9]=3
    mapping2pervideo[8]=3
    mapping2pervideo[7]=3
    mapping2pervideo[6]=3
    mapping2pervideo[5]=3
    mapping2pervideo[4]=3
    mapping2pervideo[3]=2
    mapping2pervideo[2]=1
    
    for i in range(len(train_df)):
        vid = train_df.iloc[i,0]
        folder = train_df.iloc[i,4]
        label = train_df.iloc[i,1]
        fn = train_df.iloc[i,5]
        if fn > 15:
            fn = 15
        step = mapping2pervideo[fn]
        # step = fn//number_of_image_per_video_local
        count = 0
        for s in range(0,fn,step):
            count += 1
            file = vid + "_"+ str(s)+".jpg"
            tmp_str = os.path.join(folder,label,file)
            filenmes.append(tmp_str)
            labels.append(label)
            if count >= number_of_image_per_video_local:
                break
        if count < number_of_image_per_video_local:
            print("error, frame number is less than "+ str(number_of_image_per_video_local))
    dict_assign = {'filename': filenmes, 'label': labels}
    df = pd.DataFrame(dict_assign, columns = ['filename', 'label'])
    return df

def random_int_arr(random_number, arr):
    out = resample(arr, n_samples=random_number, replace=False, random_state=100)
    tmp = set(out)
    count = 0
    while count <len(arr):
        if arr[count] in tmp:
            arr.pop(count)
            count -= 1
        count += 1
    
    return out

folder = "./../../dataset/fb_db"
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
# df['frame_num']=[15 for i in range(len(df))]
print("check frame number")
# check_frame_number(folder, df)
# df.sort_values('filename').to_csv('dataset_orginal.csv', index=False)
df = pd.read_csv("./dataset_orginal.csv")

# loop_for_reorder(df, folder)

print(" Sample dataframe")
existed_arr = [i for i in range(50)]
train_arr = random_int_arr(train_folder_number, existed_arr)
train_folder_list = create_folder_list(train_arr)
valid_arr = random_int_arr(valid_folder_number, existed_arr)
valid_folder_list = create_folder_list(valid_arr)
test_arr = random_int_arr(test_folder_number, existed_arr)
test_folder_list = create_folder_list(test_arr)

print("sample for training")
loop_for_sample_data(train_folder_list, "train", df)
print("sample for validation")
loop_for_sample_data(valid_folder_list, "valid", df)
print("sample for testing")
loop_for_sample_data(test_folder_list, "test", df)

df.sort_values('filename').to_csv('dataset_vid_2.csv', index=False)


# Create list of files for training
train_df = df[df['split']=='train']
print("create output file for training")
out_train_df = create_training_file(train_df)
out_train_df.to_csv('training_dataset_2.csv', index=False)

valid_df = df[df['split']=='valid']
print("create output file for validation")
out_valid_df = create_training_file(valid_df)
out_valid_df.to_csv('valid_dataset_2.csv', index=False)

test_df = df[df['split']=='test']
print("create output file for testing")
out_test_df = create_training_file(test_df)
out_test_df.to_csv('test_dataset_2.csv', index=False)








