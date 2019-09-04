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


save_mode = 2
epochs = 10

list_id = '04'

dir_base = '~/CAD_120'
dir_save = os.path.join(dir_base, 'models_vgg')

obj_list = load_words(os.path.join(dir_base, 'knowledge/object_list.txt'))
attr_list = load_words(os.path.join(dir_base, 'knowledge/attribute_list.txt'))
rel_list = load_words(os.path.join(dir_base, 'knowledge/relation_list.txt'))
act_list = load_words(os.path.join(dir_base, 'knowledge/action_list.txt'))

file_name_anno = os.path.join(dir_base, 'annotations/states/rel_'+list_id+'.json')
dir_video = os.path.join('~/CAD_120/videos')

with open(file_name_anno, 'r') as f:
    data = json.load(f)    

num_all = 0
for rel in rel_list:
    num_all = num_all+len(data[rel])
    print (rel, len(data[rel])) 


sample_num = 30000

data_anno = []
class_weight = {}    
for k in range(len(rel_list)):
    rel = rel_list[k]
    num = min(sample_num, len(data[rel]))
    class_weight[k] = num_all/num    
   
    print (rel, num)
    
    ids = list(range(0, len(data[rel])))
    sample_ids = random.sample(ids, num)
    for k in sample_ids:
        data_anno.append(data[rel][k])


def get_model_rel(vgg16, len_obj_list, len_rel_list, dropout = 0.5):
    
    # appearace feature from vgg16 model
    fc1_img = Dense(4096, activation = 'relu', name = 'fc1_img')(vgg16.layers[-4].output)
    fc1_img = Dropout(dropout)(fc1_img)
    fc2_img = Dense(4096, activation = 'relu', name = 'fc2_img')(fc1_img)
    fc2_img = Dropout(dropout)(fc2_img)
        
    prob_rel = Dense(len_rel_list, activation='softmax', name = 'prob_rel')(fc2_img)

    model_rel = Model(inputs = [vgg16.input], outputs = prob_rel)
    
    return model_rel


batch_size = 16
learning_rate = 1e-5
dropout = 0.5

with tf.device('/gpu:0'):
    
    gpu_option = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option, allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())    

    vgg16 = VGG16(weights='imagenet')

    model_rel = get_model_rel(vgg16, len(obj_list), len(rel_list), dropout)


adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
model_rel.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])

for epoch in range(epochs) :
    #train_data = sample_train_data_rel(data, rel_list, sample_num)  #training with all data
            
    train_data = []
    ids = list(range(0, len(data_anno)))
    random.shuffle(ids)
    for k in ids:
        train_data.append(data_anno[k])        
    
    batch_union_img = np.empty([0, 224, 224, 3])
    batch_rel_label = np.empty([0, len(rel_list)])    
    
    for k in range(len(train_data)):
        person_id = train_data[k]['person_id']
        video_label = train_data[k]['video_label']
        video_id = train_data[k]['video_id']
        obj_labels = train_data[k]['obj_labels']
        frame_id = train_data[k]['frame_id']
        rois = train_data[k]['rois']
        rel_label = train_data[k]['rel_label']

        # get roi_img image
        dir_img = os.path.join(dir_video, person_id, video_label, video_id)  
        img = image.load_img(os.path.join(dir_img, 'RGB_' + str(frame_id + 1) + '.png'))
        
        union_img = get_union_img(img, rois, 5)         
        rel_label = word2vec(rel_label, rel_list)
        
        batch_union_img = np.append(batch_union_img, [union_img], axis=0)
        batch_rel_label = np.append(batch_rel_label, [rel_label], axis=0)
        
        if batch_union_img.shape[0] == batch_size or k == len(train_data)-1:
            train = model_rel.train_on_batch([batch_union_img], batch_rel_label, class_weight = class_weight)
                        
            print ('Epoch = '+str(epoch+1)+'/'+str(epochs)+',  Progress = '+str(k+1)+'/'+str(len(train_data)), train)

            batch_union_img = np.empty([0, 224, 224, 3])
            batch_rel_label = np.empty([0, len(rel_list)])    
    
    if (epoch+1)%save_mode == 0:
        model_save_name = os.path.join(dir_save, 'model_rel_'+list_id+'_'+str(epoch+1)+'.h5') 
        model_rel.save(model_save_name)

