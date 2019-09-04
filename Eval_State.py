
import os
import sys
sys.path.insert(0, 'utils')
import numpy as np
import math
import json

from process import *
from utils import *

np.random.seed(123)


dir_base = '~/CAD_120'

dir_anno = os.path.join(dir_base, 'annotations/all')
file_name_video = os.path.join(dir_base, 'annotations/video_clips.txt')

obj_list = load_words(os.path.join(dir_base, 'knowledge/object_list.txt'))
attr_list = load_words(os.path.join(dir_base, 'knowledge/attribute_list.txt'))
rel_list = load_words(os.path.join(dir_base, 'knowledge/relation_list.txt'))
act_list = load_words(os.path.join(dir_base, 'knowledge/action_list.txt'))
obj_rel_list = load_anno_list(os.path.join(dir_base, 'knowledge/object_relation_list.txt'))


list_id = '04'

dir_result_attr = os.path.join(dir_base, 'results', list_id)
dir_result_rel = os.path.join(dir_base, 'results', list_id)

file_name_train = os.path.join(dir_base, 'splits/testlist'+list_id+'.txt')

with open(file_name_video, 'r') as f:
    video_list = [x.replace('\n', '') for x in f.readlines()]

with open(file_name_train, 'r') as f:
    test_list = [x.replace('\n', '') for x in f.readlines()]

video_list_test = []   
for video in video_list:
    items = video.replace(' ', '').split(',')
    if items[0] in test_list: 
        video_list_test.append(video)
        
print (len(video_list_test))       
print (video_list_test[0])


acc_list_attr = []
acc_list_rel = []

for k in range(0, len(video_list_test)):
    video = video_list_test[k]
    items = video.replace(' ', '').split(',')
    person_id = items[0]
    video_label = items[1]
    video_id = items[2]
    id1 = int(items[3])
    id2 = int(items[4])
    
    file_name_json = os.path.join(dir_anno, person_id, video_label, video_id+'.json')    
    with open(file_name_json, 'r') as f:
        data_anno = json.load(f)
    
    attributes_anno = data_anno['attributes']
    relations_anno = data_anno['relations']    
    
    file_attr = os.path.join(dir_result_attr, person_id, video_label, video_id, str(id1)+'_'+str(id2)+'_attr.json')
    file_rel = os.path.join(dir_result_rel, person_id, video_label, video_id, str(id1)+'_'+str(id2)+'_rel.json')    

    with open(file_attr, 'r') as f:
        attributes_test = json.load(f)
           
    with open(file_rel, 'r') as f:
        relations_test = json.load(f)        
             
            
    # evaluate attribute 
    for i in range(len(attributes_anno)):
        obj_id_anno = attributes_anno[i]['obj_id']
        attr_list_anno = attributes_anno[i]['attr_list']
             
        for j in range(len(attributes_test)):
            obj_id_test = attributes_test[j]['obj_id']
            if obj_id_anno == obj_id_test:
                attr_list_test = attributes_test[j]['attr_list']
                break
                
        for j in range(id1-1, id2):            
            if '-1.0' not in attr_list_test[j-id1+1]:  # the object is not occluded 
                attr_id = np.argmax(np.array(attr_list_test[j-id1+1], np.float32))
                attr_test = attr_list[attr_id]
                if attr_list_anno[j] != attr_test:
                    acc_list_attr.append([attr_list_anno[j], 0])
                else:
                    acc_list_attr.append([attr_list_anno[j], 1])
    
    # evaluate relation 
    for i in range(len(relations_anno)):
        obj_ids_anno = relations_anno[i]['obj_ids']
        rel_list_anno = relations_anno[i]['rel_list']
                        
        for j in range(len(relations_test)):
            obj_ids_test = relations_test[j]['obj_ids']                       
            if obj_ids_anno == obj_ids_test:
                rel_list_test = relations_test[j]['rel_list']
                break
                
        for j in range(id1-1, id2):
            if '-1.0' not in rel_list_test[j-id1+1]:  # the object is not occluded 
                rel_id = np.argmax(np.array(rel_list_test[j-id1+1], np.float32))
                rel_test = rel_list[rel_id]
                if rel_list_anno[j] != rel_test:
                    acc_list_rel.append([rel_list_anno[j], 0])
                else:
                    acc_list_rel.append([rel_list_anno[j], 1])


eval_list_attr = {}
for attr in attr_list:
    eval_list_attr[attr] = []
    
for acc_attr in acc_list_attr:
    attr = acc_attr[0]
    acc = acc_attr[1]
    eval_list_attr[attr].append(acc)
    
for attr in attr_list:
    print (attr, len(eval_list_attr[attr]), sum(eval_list_attr[attr])/len(eval_list_attr[attr]))    
    

eval_list_rel = {}
for rel in rel_list:
    eval_list_rel[rel] = []
    
for acc_rel in acc_list_rel:
    rel = acc_rel[0]
    acc = acc_rel[1]
    eval_list_rel[rel].append(acc)
   
for rel in rel_list:
    print (rel, len(eval_list_rel[rel]), sum(eval_list_rel[rel])/len(eval_list_rel[rel]))    
    

