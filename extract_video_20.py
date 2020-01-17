# import numpy as np
from classifiers import *
from pipeline import *

import json
import pandas as pd

def read_existed_files(out_dirname):
    existed_filenames = set()
    for train_test in listdir(out_dirname):
        for fake_real in listdir(join(out_dirname,train_test)):
            for f in listdir(join(out_dirname,train_test,fake_real)):
                if  (f[-4:] == '.jpg'):        
                    existed_filenames.add(f.split('.')[0])
    return existed_filenames

path_json = './../fb_whole/dfdc_train_part_20/metadata.json'
dirname = './../fb_whole/dfdc_train_part_20'
out_dirname = 'fb_db'
with open(path_json) as json_file:
    data = json.load(json_file)

# converting json dataset from dictionary to dataframe
train_list = pd.DataFrame.from_dict(data, orient='index')
train_list.reset_index(level=0, inplace=True)
total_files = len(train_list)

existed_filenames = read_existed_files(out_dirname)
for i in range(1,total_files):
    vid = train_list.loc[i]['index']
    print("index : "+ str(i) + " / " + str(total_files))
    print("vid: "+vid)
    if vid.split('.')[0] in existed_filenames:
        print('Video is extracted, id: ' + vid)
        continue
    if isfile(join(dirname, vid)) and ((vid[-4:] == '.mp4') or (vid[-4:] == '.avi') or (vid[-4:] == '.mov')):
        out_filename = join(out_dirname,train_list.loc[i]['split'],train_list.loc[i]['label'])
        extract_face_from_video(dirname,vid,out_filename)