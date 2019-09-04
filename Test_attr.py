
import os
import sys
sys.path.insert(0, 'utils')
import numpy as np
import math
import json
import tensorflow as tf

import keras
from keras.utils import plot_model

from keras.models import Model, load_model
from keras.layers import Input, Dense, Multiply, Flatten, Concatenate, Dropout
from keras.preprocessing import image

from keras.applications.vgg16 import VGG16, preprocess_input

from process import *
from utils import *

np.random.seed(123)

dir_base = '~/CAD_120'

obj_list = load_words(os.path.join(dir_base, 'knowledge/object_list.txt'))
attr_list = load_words(os.path.join(dir_base, 'knowledge/attribute_list.txt'))
rel_list = load_words(os.path.join(dir_base, 'knowledge/relation_list.txt'))
act_list = load_words(os.path.join(dir_base, 'knowledge/action_list.txt'))

dir_video = os.path.join('~/CAD_120/videos')
dir_save = os.path.join(dir_base, 'results')
dir_anno =  os.path.join(dir_base, 'annotations', 'all')

file_name_video = os.path.join(dir_base, 'annotations', 'video_clips.txt')

with open(file_name_video, 'r') as f:
    video_list = [x.replace('\n', '') for x in f.readlines()]


def get_obj_rois(obj_id, locations):                
    rois = []
    for i in range(len(locations)):
        if obj_id == locations[i]['obj_id']:
            rois = locations[i]['loc_list']            
            break

    return rois    

def predict_attr(model_attr, obj_label, img, roi, obj_list):    
    obj_label = word2vec(obj_label, obj_list)
    obj_label = np.asarray([obj_label])      

    roi_img = get_roi_img(img, roi)
    roi_img = np.asarray([roi_img])

    attr_pred = model_attr.predict([obj_label, roi_img])
    
    return attr_pred


with tf.device('/gpu:0'):
    
    gpu_option = tf.GPUOptions(allow_growth=True)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option, allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())   

model_id = '4'

for test_id in range(2, 3):
    list_id = '{:02d}'
    list_id = list_id.format(test_id+1)
    
    model_name = os.path.join(dir_base, 'models', 'model_attr_'+list_id+'_'+model_id+'.h5')
    model_attr = load_model(model_name)
    
    print (model_name)        

    file_name_test = os.path.join(dir_base, 'splits', 'testlist'+list_id+'.txt') 
    with open(file_name_test, 'r') as f:
        test_list = [x.replace('\n', '') for x in f.readlines()]

    video_list_test = []   
    for video in video_list:
        items = video.replace(' ', '').split(',')
        if items[0] in test_list: 
            video_list_test.append(video)

    print (list_id, len(video_list_test), len(video_list))
    
    for k in range(len(video_list_test)):
        items = video_list_test[k].split(', ')
        person_id = items[0]
        video_label = items[1]
        video_id = items[2]
        id1 = int(items[3])
        id2 = int(items[4])

        dir_img = os.path.join(dir_video, person_id, video_label, video_id)  
        frame_list = os.listdir(dir_img) 
        
        file_rel = os.path.join(dir_anno, person_id, video_label, video_id+'.json')
        with open(file_rel, 'r') as f:
            data = json.load(f)
        
        locations = data['locations']
        attributes = data['attributes']

        results = []
        for i in range(len(attributes)):
            attr = attributes[i]
            obj_id = attr['obj_id']
            attr_list = attr['attr_list']

            items = obj_id.split('_')            
            obj_label = items[0]

            rois = get_obj_rois(obj_id, locations)    

            result_attr = {}
            result_attr['obj_id'] = obj_id
            result_attr['attr_list'] = []

            dir_img = os.path.join(dir_video, person_id, video_label, video_id)  
            for frame_id in range(id1-1, id2):
                if rois[frame_id][2]>0:
                    frame = image.load_img(os.path.join(dir_img, 'RGB_' + str(frame_id + 1) + '.png'))
                    attr_pred = predict_attr(model_attr, obj_label, frame, rois[frame_id], obj_list)

                    attr_pred_list = []
                    for n in range(attr_pred.shape[1]):                        
                        str_attr_pred = '{:.5f}'.format(attr_pred[0][n])
                        attr_pred_list.append(str_attr_pred)

                    result_attr['attr_list'].append(attr_pred_list)

                else:
                    attr_pred_list = ['-1.0', '-1.0']
                    result_attr['attr_list'].append(attr_pred_list)
                    
                print  ('Progress = ' + str(k+1) + '/' + str(len(video_list_test)) + 
                   ',  attributes = ' + str(i + 1) + '/' + str(len(attributes)) + \
                   ',  frame_id = ' + str(frame_id + 1) + '/' + str(id2))    

                    
            results.append(result_attr)

        dir_save = os.path.join(dir_base, 'results', list_id+'_'+model_id)
        dir_name_save = os.path.join(dir_save, person_id, video_label, video_id)
        file_name_save = os.path.join(dir_name_save,  str(id1)+'_'+str(id2)+ '_attr.json')
        
        if not os.path.exists(dir_name_save):
            os.makedirs(dir_name_save)

        with open(file_name_save, 'w') as f:
            json.dump(results, f)   

        with open(file_name_save, 'r') as f:
            data_test = json.load(f)
    

