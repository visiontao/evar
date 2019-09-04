import os
import sys
import numpy as np
import math

def smooth(x, bandwidth=5, window='hamming'):
    x = np.asarray(x, np.float32)
    if x.ndim != 1:
        print ("smooth only accepts 1 dimension arrays.")
    if x.size < bandwidth:
        print ("Input vector needs to be bigger than window size.")
    if bandwidth < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        print ("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[2 * x[0] - x[bandwidth - 1::-1], x, 2 * x[-1] - x[-1:-bandwidth:-1]]
    if window == 'flat':  # moving average
        w = np.ones(bandwidth, 'd')
    else:
        w = eval('np.' + window + '(bandwidth)')
    y = np.convolve(w / w.sum(), s, mode='same')
    
    return y[bandwidth:-bandwidth + 1] 

def smooth_state_list(state_list, bandwidth):
    result_list = np.zeros([len(state_list), len(state_list[0])], np.float32)
    for i in range(len(state_list[0])):
        temporal_list = []
        for j in range(len(state_list)):
            temporal_list.append(state_list[j][i])

        smooth_list = smooth(temporal_list, bandwidth)
        for j in range(smooth_list.size):
            result_list[j][i] = smooth_list[j]

    return result_list

def get_continuous_state(state_list, bandwidth):
    frame_neighbor = int(0.5*(bandwidth+1)) 
    # detect the state is continuous or not
    flag_list = []
    for i in range(len(state_list)):
        flag = False
        for j in range(-frame_neighbor+1, 1):
            id1 = max(0, i+j)
            if id1+frame_neighbor < len(state_list):
                state_set = set(state_list[id1:id1+frame_neighbor])    
                if len(state_set) == 1:
                    flag = True
                    break
        flag_list.append(flag)
    
    return flag_list

def get_continuous_id(flag_list):
    # find the continuous frame id    
    continuous_ids = []
    for i in range(len(flag_list)):
        if flag_list[i] == True:
            continuous_ids.append(i)
            
    return continuous_ids

def rectify_nearest_state(state_list, idx, bandwidth):
    # rectify one value 
    rectified_list = state_list.copy()
    
    flag_list = get_continuous_state(state_list, bandwidth)        
    continuous_ids = get_continuous_id(flag_list)
    
    # find the nearest continuous state
    for i in range(idx, len(state_list)):
        if flag_list[i] == False:
            frame_dist = []
            for j in range(0, len(continuous_ids)):
                dist = abs(i-continuous_ids[j])
                frame_dist.append(dist)

            if len(frame_dist) > 1:
                min_idx = frame_dist.index(min(frame_dist))
                id_nearest = continuous_ids[min_idx]
            else:
                id_nearest = 0
            rectified_list[i] = state_list[id_nearest]  
            break
            
    return rectified_list

def rectify_state_list(state_list, bandwidth):
    rectify_list = state_list.copy()
    for i in range(len(rectify_list)):
        rectify_list = rectify_nearest_state(rectify_list, i, bandwidth)
        
    return rectify_list

def remove_occlusion(state_list):
    result_list = state_list.copy()
    for i in range(len(state_list)-1, -1, -1):
        if state_list[i] == -1:
            result_list.remove(-1)
            
    return result_list

def remove_occlusion(state_list):
    result_list = state_list.copy()
    for i in range(len(state_list)-1, -1, -1):
        if state_list[i] == -1:
            result_list.remove(-1)
            
    return result_list

def get_prob_rel_list(obj_labels, rel_list, obj_rel_list):
    for obj_rel in obj_rel_list:
        if obj_labels[0] == obj_rel[0]:
            objs = obj_rel[3].split(',')            
            if obj_labels[1] in objs:                
                id1 = word2index(obj_rel[1], rel_list)
                id2 = word2index(obj_rel[2], rel_list)
                break
                
    prob_rel_list = []
    for i in range(len(rel_list)):
        if i == id1 or i == id2:
            prob_rel_list.append(1)
        else:
            prob_rel_list.append(0)
                
    return prob_rel_list

def word2index(word, words):
    id = [i for i,x in enumerate(words) if x == word][0]
    return id

def get_prob_rel_vec(obj_labels, rel_list, obj_rel_list):
    prob_rel = np.zeros(len(rel_list), dtype = np.float32)
    for obj_rel in obj_rel_list:
        if obj_labels[0] == obj_rel[0]:
            objs = obj_rel[3].split(',')            
            if obj_labels[1] in objs:                
                id1 = word2index(obj_rel[1], rel_list)
                id2 = word2index(obj_rel[2], rel_list)
                prob_rel[id1] = 1.0
                prob_rel[id2] = 1.0
                break
    return prob_rel
