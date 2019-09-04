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
from keras.layers import Input, Dense, Multiply, Flatten, Concatenate

from process import *
from utils import *

np.random.seed(123)


dir_base = '~/CAD_120'

obj_list = load_words(os.path.join(dir_base, 'knowledge/object_list.txt'))
attr_list = load_words(os.path.join(dir_base, 'knowledge/attribute_list.txt'))
rel_list = load_words(os.path.join(dir_base, 'knowledge/relation_list.txt'))
act_list = load_words(os.path.join(dir_base, 'knowledge/action_list.txt'))

obj_attr_list = load_anno_list(os.path.join(dir_base, 'knowledge/object_attribute_list.txt'))
obj_rel_list = load_anno_list(os.path.join(dir_base, 'knowledge/object_relation_list.txt'))
act_rel_list = get_text_line(os.path.join(dir_base, 'knowledge/act_rel_list.txt'))


# generate data for relation-based action recognition
def get_state_act_rel(obj_list, obj_rel_list, act_rel_list):
    data_state = []
    
    # get labeled action data
    labeled_state_list = []
    for act_rel in act_rel_list:
        line = act_rel.split(';') 
        act = line[0]
        rel_pre = line[1]
        rel_eff = line[2]

        if len(line) == 4:
            objs = line[3].split(',')
            for obj in objs:            
                data_state.append([obj, obj, rel_pre, rel_eff, act])
                labeled_state_list.append([obj, obj, rel_pre, rel_eff])

        if len(line) == 5:
            subs = line[3].split(',')
            objs = line[4].split(',')
            for sub in subs:
                for obj in objs:
                    data_state.append([sub, obj, rel_pre, rel_eff, act])
                    labeled_state_list.append([sub, obj, rel_pre, rel_eff])


    # generate unlabeled 'null' action
    for sub in obj_list:
        for obj in obj_list:
            for rel_pre in rel_list:
                for rel_eff in rel_list:
                    flag_pre = is_concerned_rel([sub, obj], rel_pre, obj_rel_list)
                    flag_eff = is_concerned_rel([sub, obj], rel_eff, obj_rel_list)
                    if flag_pre and flag_eff:                
                        state = [sub, obj, rel_pre, rel_eff]                        
                        if state not in labeled_state_list:
                            data_state.append([sub, obj, rel_pre, rel_eff, 'null'])
                            
    return data_state



def get_data_act_rel(data_state, obj_list, attr_list, act_list):
    sub_labels = np.empty([0, len(obj_list)])
    obj_labels = np.empty([0, len(obj_list)])
    rel_pres = np.empty([0, len(rel_list)])
    rel_effs = np.empty([0, len(rel_list)])
    act_labels = np.empty([0, len(act_list)])

    for state in data_state:
        sub_label = word2vec(state[0], obj_list)
        sub_labels = np.append(sub_labels, [sub_label], axis=0)

        obj_label = word2vec(state[1], obj_list)
        obj_labels = np.append(obj_labels, [obj_label], axis=0)

        rel_pre = word2vec(state[2], rel_list)
        rel_pres = np.append(rel_pres, [rel_pre], axis=0)
        rel_eff = word2vec(state[3], rel_list)
        rel_effs = np.append(rel_effs, [rel_eff], axis=0)

        act_label = word2vec(state[4], act_list)
        act_labels = np.append(act_labels, [act_label], axis=0)
        
    
    data_train = [sub_labels, obj_labels, rel_pres, rel_effs, act_labels]
        
    return data_train


# action model with relationship changes

def get_model_act_rel(len_obj_list, len_rel_list, len_act_list):
    sub_label = Input(shape=(len_obj_list,))
    obj_label = Input(shape=(len_obj_list,))
    rel_pre = Input(shape=(len_rel_list,))
    rel_eff = Input(shape=(len_rel_list,))

    states = Concatenate()([sub_label, obj_label, rel_pre, rel_eff])
    prob_act_rel = Dense(len(act_list), activation='softmax', name='prob_act_rel')(states)

    model_act_rel = Model(inputs=[sub_label, obj_label, rel_pre, rel_eff], outputs=prob_act_rel)
    
    return model_act_rel



data_state = get_state_act_rel(obj_list, obj_rel_list, act_rel_list)
data_train = get_data_act_rel(data_state, obj_list, attr_list, act_list)

print (len(data_state))

with tf.device('/gpu:0'):
    
    gpu_option = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option, allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())    

    model_act_rel = get_model_act_rel(len(obj_list), len(rel_list), len(act_list))
    
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
    model_act_rel.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
    

    model_act_rel.fit([data_train[0], data_train[1], data_train[2], data_train[3]], data_train[4], 
            validation_data = ([data_train[0], data_train[1], data_train[2], data_train[3]], data_train[4]),
                       batch_size = 8, epochs = 100, verbose = 1)
    
    model_act_rel.save(os.path.join(dir_base, 'models/model_act_rel.h5'))
    del model_act_rel

