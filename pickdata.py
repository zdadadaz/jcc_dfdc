from classifiers import *
from pipeline import *

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os

def read_existed_files(out_dirname):
    existed_filenames = set()
    for train_test in listdir(out_dirname):
        for fake_real in listdir(join(out_dirname,train_test)):
            for f in listdir(join(out_dirname,train_test,fake_real)):
                if  (f[-4:] == '.jpg'):        
                    existed_filenames.add(f.split('.')[0])
    return existed_filenames

def read_json_files(dir_json):
    all_files = []
    for js in listdir(dir_json):
        if  (js[-5:] == '.json'):
            with open(join(dir_json,js)) as json_file:
                data = json.load(json_file)
                train_list = pd.DataFrame.from_dict(data, orient='index')
                train_list.reset_index(level=0, inplace=True)
                if len(all_files) ==0:
                    all_files = train_list
                else:
                    all_files = all_files.append(train_list,ignore_index = True)
    return all_files

def move_image(df, dir_in, dir_out, image_dict, isReal, isTrain):
    for i in range(len(df)):
        vid = df.iloc[i][0].split(".")[0]
        if vid not in image_dict:
            continue
        else:
            preReal = "REAL" if isReal else "FAKE"
            preTrain = "train" if isTrain else "val"
            cmd = 'ls -1 ' + join(dir_in,"train",preReal, vid+"*")
            tmp= os.popen(cmd).read()
            files = tmp.split("\n")
            files = files[:-1] if len(files[-1]) == 0 else files
            step = 1 if len(files) < 6 else int(len(files)/6)
            start = 0 if len(files) == 1 else 1
            for f in range(start,len(files),step):
                cmd2 = 'cp ' + files[f] +" " + join(dir_out, preTrain, preReal,files[f].split("/")[-1])
                os.system(cmd2)
            

fb_dirname = 'fb_db'
out_dirname = 'db_small'
path_json = './../fb_whole/'
files = read_json_files(path_json)
images = read_existed_files(fb_dirname)
real = files[files['label']=='REAL']
# real = resample(files[files['label']=='REAL'])
train_real, test_real = train_test_split(real, test_size=0.2, random_state=200)
fake = resample(files[files['label']=='FAKE'], n_samples=len(real), replace=False, random_state=800)
train_fake, test_fake = train_test_split(fake, test_size=0.2,random_state=1000)
move_image(train_real, fb_dirname, out_dirname,images, True, True)
move_image(train_fake, fb_dirname, out_dirname,images, False, True)
move_image(test_real, fb_dirname, out_dirname,images, True, False)
move_image(test_fake, fb_dirname, out_dirname,images, False, False)
