
import os
import sys
sys.path.insert(0, 'net')
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
from eval_act import *

np.random.seed(123)



dir_base = '~/CAD_120'
dir_anno = os.path.join(dir_base, 'annotations/all')
dir_video = os.path.join('~/CAD_120/videos')

obj_list = load_words(os.path.join(dir_base, 'knowledge/object_list.txt'))
attr_list = load_words(os.path.join(dir_base, 'knowledge/attribute_list.txt'))
rel_list = load_words(os.path.join(dir_base, 'knowledge/relation_list.txt'))
act_list = load_words(os.path.join(dir_base, 'knowledge/action_list.txt'))
obj_rel_list = load_anno_list(os.path.join(dir_base, 'knowledge/object_relation_list.txt'))



file_name_video = os.path.join(dir_base, 'annotations/video_clips.txt')


model_id = '10'
list_id = '02'

dir_result_attr = os.path.join(dir_base, 'results', list_id)
dir_result_rel = os.path.join(dir_base, 'results', list_id)

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


print (len(video_list_test), test_list[0])


# In[17]:


# load state of the video

result_video = []
for k in range(0, len(video_list_test)):
    items = video_list_test[k].split(', ')

    person_id = items[0]
    video_label = items[1]
    video_id = items[2]   
    
    id1 = int(items[3])
    id2 = int(items[4])

    file_attr = os.path.join(dir_result_attr, person_id, video_label, video_id, str(id1)+'_'+str(id2)+'_attr.json')
    file_rel = os.path.join(dir_result_rel, person_id, video_label, video_id, str(id1)+'_'+str(id2)+'_rel.json')
    
    with open(file_attr, 'r') as f:
        state_attr = json.load(f)
           
    with open(file_rel, 'r') as f:
        state_rel = json.load(f)
                
    state_clip = {}
    state_clip['video'] = video_list_test[k]
    state_clip['attributes'] = []
    state_clip['relations'] = []
  
    for i in range(0, len(state_attr)):
        state = state_attr[i]
        obj_id = state['obj_id']
        attributes = state['attr_list']
            
        state_obj_attr = {}
        state_obj_attr['obj_id'] = state['obj_id']
        state_obj_attr['attr_list'] = []
        
        for j in range(len(attributes)):
            prob_attr = np.array(attributes[j], np.float32)                        
            if np.min(prob_attr) == -1:
                attr_label = -1
            else:
                attr_label = np.argmax(prob_attr)   
                
            state_obj_attr['attr_list'].append(attr_label)

        state_clip['attributes'].append(state_obj_attr)
        
    
    for i in range(len(state_rel)):
        state = state_rel[i]
        obj_ids = state['obj_ids'] 
        relations = state['rel_list'] 
        
        sub_items = obj_ids[0].split('_')
        obj_items = obj_ids[1].split('_')
        
        sub_label = sub_items[0]
        obj_label = obj_items[0]
        prob_obj_rel = get_prob_rel_vec([sub_label, obj_label], rel_list, obj_rel_list)   
        
        state_obj_rel = {}
        state_obj_rel['obj_ids'] = obj_ids
        state_obj_rel['rel_list'] = []
        
        for j in range(len(relations)):
            prob_rel = np.array(relations[j], np.float32)                        
            if np.min(prob_rel) == -1:
                rel_label = -1
            else:
                prob_rel = np.array(relations[j], np.float32)*prob_obj_rel      
                if sum(prob_rel) == 0:
                    rel_label = -1
                else:  
                    rel_label = np.argmax(prob_rel)    
                
            state_obj_rel['rel_list'].append(rel_label)

        state_clip['relations'].append(state_obj_rel)
     
    result_video.append(state_clip)


with tf.device('/gpu:0'):

    gpu_option = tf.GPUOptions(allow_growth=True)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option, allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())     

    bandwidth = 1
    video_act_list = []

    model_act_attr = load_model(os.path.join(dir_base, 'models/model_act_attr.h5'))
    model_act_rel = load_model(os.path.join(dir_base, 'models/model_act_rel.h5'))

    for k in range(0, len(result_video)):       
        video = result_video[k]['video']
        state_attr = result_video[k]['attributes']
        state_rel = result_video[k]['relations']

        video_act = {}
        video_act['video'] = video
        video_act['attributes'] = []
        video_act['relations'] = []       

        # attribute-based action
        for i in range(0, len(state_attr)):
            obj_id = state_attr[i]['obj_id']
            state_list_attr = state_attr[i]['attr_list']       
            state_list_attr = remove_occlusion(state_list_attr)

            if len(state_list_attr) > bandwidth:        
                state_list_attr = rectify_state_list(state_list_attr, bandwidth)

                state_set_attr = []
                for attr in state_list_attr:
                    if attr not in state_set_attr:
                        state_set_attr.append(attr)
                    elif attr != state_set_attr[-1]:
                        state_set_attr.append(attr)

                if len(state_set_attr)>1:     
                    obj_items = obj_id.split('_')
                    obj_label = obj_items[0]                
                    obj_label = word2vec(obj_label, obj_list)
                    obj_label = np.asarray([obj_label])                

                    for j in range(1, len(state_set_attr)):
                        attr_pre = attr_list[state_set_attr[j-1]]   
                        attr_pre = word2vec(attr_pre, attr_list)            
                        attr_pre = np.asarray([attr_pre])

                        attr_eff = attr_list[state_set_attr[j]]   
                        attr_eff = word2vec(attr_eff, attr_list)            
                        attr_eff = np.asarray([attr_eff])

                        act_attr_pred = model_act_attr.predict([obj_label, attr_pre, attr_eff])   
                        act_pred = np.argmax(act_attr_pred)

                        video_act['attributes'].append([obj_id, act_pred])


        # relation-based action
        for i in range(0, len(state_rel)):
            obj_ids = state_rel[i]['obj_ids']
            state_list_rel = state_rel[i]['rel_list']       
            state_list_rel = remove_occlusion(state_list_rel)

            if len(state_list_rel) > bandwidth:        
                state_list_rel = rectify_state_list(state_list_rel, bandwidth)

                state_set_rel = []
                for rel in state_list_rel:
                    if rel not in state_set_rel:
                        state_set_rel.append(rel)
                    elif rel != state_set_rel[-1]:
                        state_set_rel.append(rel)

                if len(state_set_rel)>1:                            
                    sub_items = obj_ids[0].split('_')
                    sub_label = sub_items[0]
                    sub_label = word2vec(sub_label, obj_list)
                    sub_label = np.asarray([sub_label])

                    obj_items = obj_ids[1].split('_')
                    obj_label = obj_items[0]
                    obj_label = word2vec(obj_label, obj_list)
                    obj_label = np.asarray([obj_label])                

                    for j in range(1, len(state_set_rel)):
                        rel_pre = rel_list[state_set_rel[j-1]]
                        rel_pre = word2vec(rel_pre, rel_list)
                        rel_pre = np.asarray([rel_pre])      

                        rel_eff = rel_list[state_set_rel[j]]
                        rel_eff = word2vec(rel_eff, rel_list)            
                        rel_eff = np.asarray([rel_eff])

                        act_rel_pred = model_act_rel.predict([sub_label, obj_label, rel_pre, rel_eff])   
                        act_pred = np.argmax(act_rel_pred)

                        video_act['relations'].append([obj_ids, act_pred])

        video_act_list.append(video_act)


eval_list = {}
for act in act_list:
    eval_list[act] = []

video_list_error = []
for k in range(len(video_act_list)):
    video_act = video_act_list[k]
    video = video_act['video']
    act_list_attr = video_act['attributes']
    act_list_rel = video_act['relations']
        
    items = video.replace(' ','').split(',')
    act_gt = word2index(items[-1], act_list)     
    
    # collect the detected actions 
    act_list_pred = []
    for act_attr in act_list_attr:
        act_list_pred.append(act_attr[1])
    
    for act_rel in act_list_rel:
        act_list_pred.append(act_rel[1])
            
    # remove the null actions 
    for i in range(len(act_list)):
        if 0 in act_list_pred:
            act_list_pred.remove(0)    
    
    pos = 0
    neg = 0
    for act in act_list_pred:
        if act != act_gt:
            neg = neg+1
        else:
            pos = pos+1     
    
    acc = 0
    if act_gt == 0:
        if len(act_list_pred) > 0:
            acc = 0
        else:
            acc = 1
    else:
        if pos+neg > 0:
            acc = pos / (pos+neg)
            
        
    act_label = act_list[act_gt]
    eval_list[act_label].append(acc)
    
    if acc != 1:
        video_list_error.append([k, act_gt, video_act])
       

avg_rec = 0
avg_acc = 0
recall_list = []
for act in act_list:
    num = len(eval_list[act])
    if num > 0:
        recall = 0
        for k in range(num):
            if eval_list[act][k] > 0:
                recall = recall+1            
        recall = recall / num    
        accuracy = sum(eval_list[act])/num
    else:
        recall = 0
        accuracy = 0

    avg_rec = avg_rec+recall*num
    avg_acc = avg_acc+accuracy*num
    
    recall_list.append(recall)
    
    print (act, num, recall, accuracy)

avg_rec = avg_rec/len(video_act_list)
avg_acc = avg_acc/len(video_act_list)
    
print ()
print (len(video_list_test))
print (avg_rec, avg_acc)

print (recall_list)
print (np.mean(recall_list))


