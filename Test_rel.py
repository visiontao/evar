
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
dir_anno =  os.path.join(dir_base, 'annotations', 'all')

file_name_video = os.path.join(dir_base, 'annotations', 'video_clips.txt')

def get_obj_rois(obj_id, locations):                
    rois = []
    for i in range(len(locations)):
        if obj_id == locations[i]['obj_id']:
            rois = locations[i]['loc_list']            
            break

    return rois   

def predict_rel(model_rel, obj_labels, img, rois, obj_list, margin = 5):
    sub_label = word2vec(obj_labels[0], obj_list)
    sub_label = np.asarray([sub_label])

    obj_label = word2vec(obj_labels[1], obj_list)
    obj_label = np.asarray([obj_label])      
    
    union_img = get_union_img(img, rois, margin)                      
    union_img = np.asarray([union_img])
    
    rel_pred = model_rel.predict([sub_label, obj_label, union_img])
    
    return rel_pred


with tf.device('/gpu:0'):

    gpu_option = tf.GPUOptions(allow_growth=True)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option, allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())     


model_id = '4'    

for test_id in range(2, 3):
    list_id = '{:02d}'.format(test_id+1) 
    print (list_id)

    model_rel = load_model(os.path.join(dir_base, 'models', 'model_rel_'+list_id+'_'+model_id+'.h5'))
    file_name_test = os.path.join(dir_base, 'splits', 'testlist'+list_id+'.txt') 
    
    with open(file_name_video, 'r') as f:
        video_list = [x.replace('\n', '') for x in f.readlines()]

    with open(file_name_test, 'r') as f:
        test_list = [x.replace('\n', '') for x in f.readlines()]

    video_list_test = []   
    for video in video_list:
        items = video.replace(' ', '').split(',')
        if items[0] in test_list: 
            video_list_test.append(video)

    print (list_id, len(video_list_test), len(video_list))
            
    for k in range(0, len(video_list_test)):
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
        relations = data['relations']    
        
        results = []
        for i in range(0, len(relations)):
            rel = relations[i]
            obj_ids = rel['obj_ids']

            items_sub = obj_ids[0].split('_')
            items_obj = obj_ids[1].split('_')
            obj_labels = [items_sub[0], items_obj[0]]
            
            sub_rois = get_obj_rois(obj_ids[0], locations)
            obj_rois = get_obj_rois(obj_ids[1], locations)
            
                            
            # saving result
            result_rel = {}
            result_rel['obj_ids'] = obj_ids
            result_rel['rel_list'] = []

            # predict relationship 
            for frame_id in range(id1-1, id2):                      
                if sub_rois[frame_id][2] > 0 and obj_rois[frame_id][2] > 0:
                    rois = [sub_rois[frame_id], obj_rois[frame_id]]
                    img = image.load_img(os.path.join(dir_img, 'RGB_' + str(frame_id + 1) + '.png'))
                    rel_pred = predict_rel(model_rel, obj_labels, img, rois, obj_list, 5)

                    rel_pred_list = []
                    for n in range(rel_pred.shape[1]):                             
                        str_rel_pred = '{:.5f}'.format(rel_pred[0][n])
                        rel_pred_list.append(str_rel_pred)

                    result_rel['rel_list'].append(rel_pred_list)
                else:
                    rel_pred_list = []
                    for rel in rel_list:
                        rel_pred_list.append(str(-1.0))

                            
                result_rel['rel_list'].append(rel_pred_list)

                print  ('Progress = ' + str(k+1) + '/' + str(len(video_list_test)) + 
                       ',  Relation = ' + str(i + 1) + '/' + str(len(relations)) + \
                       ',  Frame = ' + str(frame_id + 1) + '/' + str(id2))


            results.append(result_rel)      
            
        dir_save = os.path.join(dir_base, 'results', list_id+'_'+model_id)
        dir_name_save = os.path.join(dir_save, person_id, video_label, video_id)
        file_name_save = os.path.join(dir_name_save,  str(id1)+'_'+str(id2)+ '_rel.json')

        if not os.path.exists(dir_name_save):
            os.makedirs(dir_name_save)

        with open(file_name_save, 'w') as f:
            json.dump(results, f)       
        




