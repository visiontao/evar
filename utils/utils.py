import os
import sys
import math
import random

from process import *

def get_file_names(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dir_names, file_names).
    """
    file_names = []  # List which will store all of the full file_names.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for file_path in files:
            # Join the two strings in order to form the full filepath.
            file_name = os.path.join(root, file_path)
            file_names.append(file_name)  # Add it to the list.

    return file_names  

def sample_train_data_attr(data, attr_list, sample_num):

    # get the numbers of each attribute data
    attr_nums = []
    for attr in attr_list:
        attr_nums.append(len(data[attr]))
    sample_num = min(min(attr_nums), sample_num) 
        
    # sample data for training
    data_attr = []
    for attr in attr_list:
        ids = list(range(0, len(data[attr])))
        sample_ids = random.sample(ids, sample_num)

        for k in sample_ids:
            data_attr.append(data[attr][k])
    
    train_data = []
    ids = list(range(0, len(data_attr)))
    random.shuffle(ids)
    for k in ids:
        train_data.append(data_attr[k])

    return train_data

def sample_train_data_rel(data, rel_list, sample_num):

    sample_num_list = []
    for i in range(0, len(rel_list), 2):
        num1 = len(data[rel_list[i]])
        num2 = len(data[rel_list[i+1]])

        num_min = min(num1, num2, sample_num)
        sample_num_list.append(num_min)
        sample_num_list.append(num_min)

    # sample data for training
    data_rel = []
    for i in range(0, len(rel_list)):
        ids = list(range(0, len(data[rel_list[i]])))
        sample_ids = random.sample(ids, sample_num_list[i])

        for k in sample_ids:
            data_rel.append(data[rel_list[i]][k])

    train_data = []
    ids = list(range(0, len(data_rel)))
    random.shuffle(ids)
    for k in ids:
        train_data.append(data_rel[k])

    return train_data

def get_detail_data(file_name):
    with open(file_name, 'r') as f:
        lines = [x.replace('\n', '').replace('\r', '').replace(' ', '') for x in f.readlines()]

    data = []
    for line in lines:
        data_list = line.split(',')
        if len(data_list) == 3:            
            id_start = int(data_list[0])
            id_end = int(data_list[1])
            for i in range(id_start, id_end+1):
                d = data_list[2].replace('\r', '').replace(' ', '')
                data.append(d)

    return data


def get_clip_data(file_name):
    with open(file_name, 'r') as f:
        lines = [x.replace('\n', '').replace('\r', '').replace(' ', '') for x in f.readlines()]
        
    data = []
    for line in lines:
        data_list = line.split(',')
        data.append(data_list)

    return data

# obj_attr = [obj, attr1, attr2, attr3, ... ]
def is_concerned_attr(obj_label, obj_attr_list):
    flag = False
    for obj_attr in obj_attr_list:
        if obj_label == obj_attr[0]:
            if len(obj_attr[1].split(','))>1:
                flag = True
                break
                
    return flag

# obj_rel = [sub, rel1, rel2, obj1, obj2, obj3, ... ]
def is_concerned_rel(obj_labels, rel, obj_rel_list):
    flag = False
    for obj_rel in obj_rel_list:
        if obj_labels[0] == obj_rel[0]:
            objs = obj_rel[3].split(',')
            rels = [obj_rel[1], obj_rel[2]]
            if obj_labels[1] in objs and rel in rels:
                flag = True
                break
    return flag


def get_obj_labels(file_name):
    file_name = file_name.replace('.txt', '')
    path_list = file_name.split('/')
    anno_list = path_list[-1].split('_')

    if anno_list[0] == 'attr':
        obj_label = anno_list[1]
        return obj_label

    if anno_list[0] == 'rel':
        sub_label = anno_list[1]
        obj_label = anno_list[3]
        return [sub_label, obj_label]

    if anno_list[0] == 'act':
        if anno_list[1] == 'attr':
            obj_label = anno_list[2]
            return obj_label

    if anno_list[1] == 'rel':
        sub_label = anno_list[2]
        obj_label = anno_list[4]    
        return [sub_label, obj_label]


def load_location(file_name):
    bboxes = []
    with open(file_name, 'r') as f:
        #lines = [x.replace('\n', '').replace('\r', '') for x in f.readlines()]
        lines = [x.replace('\n', '') for x in f.readlines()]
        # first row of the file is object information
        for i in range(1, len(lines)):            
            data = lines[i].split(' ')
            bbox = []
            if len(data) >= 4:
                for j in range(4):
                    d = int(data[j])
                    bbox.append(d)
                bboxes.append(bbox) 
    return bboxes

def load_obj_info_list(file_name):
    obj_info_list = []
    with open(file_name, 'r') as f:
        lines = [x.replace('\n', '').replace('\r', '') for x in f.readlines()]
        # first row of the file is object information
        for line in lines:
            data_list = line.split('; ')
            obj_info_list.append(data_list)
    return obj_info_list

def load_anno_list(file_name):
    anno_list = []
    with open(file_name, 'r') as f:
        lines = [x.replace('\n', '').replace('\r', '').replace(' ', '') for x in f.readlines()]
        # first row of the file is object information
        for line in lines:
            data_list = line.split(';')
            anno_list.append(data_list)
    return anno_list

def get_text_line(file_name):
    with open(file_name, 'r') as f:
        lines = [x.replace('\n', '').replace('\r', '').replace(' ', '') for x in f.readlines()]
        
    data = []
    for line in lines:        
        data.append(line)

    return data