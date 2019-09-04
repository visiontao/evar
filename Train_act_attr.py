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
act_attr_list = get_text_line(os.path.join(dir_base, 'knowledge/act_attr_list.txt'))


def get_state_act_attr(obj_list, obj_attr_list, act_attr_list):
    # get labeled action data
    data_state = []
    
    labeled_state_list = []
    for act_attr in act_attr_list:
        line = act_attr.split(';')
        act = line[0]
        attr_pre = line[1]
        attr_eff = line[2]
        act_obj_list = line[3].split(',')

        for obj in act_obj_list:        
            data_state.append([obj, attr_pre, attr_eff, act])    
            labeled_state_list.append([obj, attr_pre, attr_eff])
            
    # generate unlabeled 'null' action
    for obj in obj_list:
        if is_concerned_attr(obj, obj_attr_list): 
            for attr_pre in attr_list:
                for attr_eff in attr_list:       
                    state = [obj, attr_pre, attr_eff]               
                    if state not in labeled_state_list:
                        data_state.append([obj, attr_pre, attr_eff, 'null'])


    return data_state



def get_data_act_attr(data_state, obj_list, attr_list, act_list):
    
    obj_labels = np.empty([0, len(obj_list)])
    attr_pres = np.empty([0, len(attr_list)])
    attr_effs = np.empty([0, len(attr_list)])
    act_labels = np.empty([0, len(act_list)])

    for state in data_state:
        obj_label = word2vec(state[0], obj_list)
        obj_labels = np.append(obj_labels, [obj_label], axis=0)

        attr_pre = word2vec(state[1], attr_list)
        attr_pres = np.append(attr_pres, [attr_pre], axis=0)
        attr_eff = word2vec(state[2], attr_list)
        attr_effs = np.append(attr_effs, [attr_eff], axis=0)

        act_label = word2vec(state[3], act_list)
        act_labels = np.append(act_labels, [act_label], axis=0)
    
    data_train = [obj_labels, attr_pres, attr_effs, act_labels]
    
    return data_train


def get_model_act_attr(len_obj_list, len_attr_list, len_act_list):
    obj_label = Input(shape=(len_obj_list,))
    attr_pre = Input(shape=(len_attr_list,))
    attr_eff = Input(shape=(len_attr_list,))
    states = Concatenate()([obj_label, attr_pre, attr_eff])
    
    print (obj_label.shape, attr_pre.shape, attr_eff.shape)
    

    prob_act_attr = Dense(len_act_list, activation = 'softmax', name = 'prob_act_attr')(states)
    model_act_attr = Model(inputs=[obj_label, attr_pre, attr_eff], outputs=prob_act_attr)

    return model_act_attr


data_state = get_state_act_attr(obj_list, obj_attr_list, act_attr_list)
data_train = get_data_act_attr(data_state, obj_list, attr_list, act_list)

print (len(data_state))
for d in data_state:
    print (d)    
    

with tf.device('/gpu:0'):    
    gpu_option = tf.GPUOptions(allow_growth=True)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option, allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())        

    model_act_attr = get_model_act_attr(len(obj_list), len(attr_list), len(act_list))

    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
    model_act_attr.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])

    model_act_attr.fit([data_train[0], data_train[1], data_train[2]], data_train[3], 
                       validation_data = ([data_train[0], data_train[1], data_train[2]], data_train[3]),
                       batch_size = 8, epochs = 300, verbose = 1)


    model_act_attr.save(os.path.join(dir_base, 'models/model_act_attr.h5'))
    del model_act_attr

