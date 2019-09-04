import numpy as np
import math

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input

def load_words(file_name):
    with open(file_name, 'r') as f:
        words = [x.replace('\n', '').replace(' ', '') for x in f.readlines()]
        return words    

def word2vec(word, words):
    vector = [0.0]*len(words)
    
    ids = [i for i,x in enumerate(words) if str(x) == str(word)]
    
    if len(ids) == 1:
        vector[ids[0]] = 1.0
    else:
        print (ids)
        print (word)
        print (words)
    
    return vector

def word2index(word, words):
    id = [i for i,x in enumerate(words) if x == word][0]
    return id

# compute relative location offset
def relative_location(bbox1, bbox2):
    dx = math.exp(-1.0 * (bbox2[0] - bbox1[0]) /  bbox1[2])
    dy = math.exp(-1.0 * (bbox2[1] - bbox1[1]) / bbox1[3])
    dw = math.exp(-1.0 * bbox2[2] / bbox1[2])
    dh = math.exp(-1.0 * bbox2[3] / bbox1[3])
    
    return [dx,dy, dw, dh]


# generate feature of roi region
def get_roi_data(img, rois, nw, nh, margin): 
    bbox = get_union_bbox(img.shape[1], img.shape[0], rois[0], rois[1], margin)
    
    roi_img = get_roi_img(img, bbox)
    dual_mask = get_dual_mask(img.shape[1], img.shape[0], rois[0], rois[1], nw, nh, margin)

    return [roi_img, dual_mask]

# generate feature of roi region
def get_union_img(img, rois, margin): 
    bbox = get_union_bbox(img.size[0], img.size[1], rois[0], rois[1], margin)    
    union_img = get_roi_img(img, bbox)    
    
    return union_img

# preprocess image
def get_roi_img(img, bbox = None):
    roi_img = np.array(img, dtype = np.float32)
    if bbox is not None:
        box = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        roi_img = img.crop(box)
        
    roi_img = roi_img.resize((224, 224))
        
    roi_img = image.img_to_array(roi_img)
    roi_img = preprocess_input(np.array(roi_img))
    
    return roi_img

# convert a mask (bw*bh) to normalized size (nw*nh)
def get_normal_mask(bw, bh, bbLoc, nw, nh):    
    rw = 1.0 * nw / bw
    rh = 1.0 * nh / bh
    x1 = max(0, int(math.floor(bbLoc[0] * rw)))
    y1 = max(0, int(math.floor(bbLoc[1] * rh)))
    x2 = min(nw, int(math.ceil(bbLoc[2] * rw)))    
    y2 = min(nh, int(math.ceil(bbLoc[3] * rh)))    
    
    mask = np.zeros((nh, nw), dtype = 'float32')
    mask[y1 : y2, x1 : x2] = 1
    
    return mask

# get bbox of union region (subject and object)
def get_union_bbox(width, height, bbox1, bbox2, margin):
    # location of union bbox region   
    x1 = max(0, min(bbox1[0], bbox2[0]) - margin)
    y1 = max(0, min(bbox1[1], bbox2[1]) - margin)
    x2 = min(width, max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2]) + margin)
    y2 = min(height, max(bbox1[1]+bbox1[3], bbox2[1] + bbox2[3]) + margin)
    
    bbox = [x1, y1, x2 - x1, y2 - y1]
    
    return bbox

# get dual region mask of union bbox (subject and object)
def get_dual_mask(width, height, bbox1, bbox2, nw, nh, margin = 0):    
    bbox = get_union_bbox(width, height, bbox1, bbox2, margin)
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] + bbox[3]
    
    # relative location of bbox1 to union bbox region
    x1_1 = bbox1[0] - x1
    y1_1 = bbox1[1] - y1
    x2_1 = bbox1[0] + bbox1[2] - x1
    y2_1 = bbox1[1] + bbox1[3] - y1

    # relative location of bbox2 to union bbox region
    x1_2 = bbox2[0] - x1
    y1_2 = bbox2[1] - y1
    x2_2 = bbox2[0] + bbox2[2] - x1
    y2_2 = bbox2[1] + bbox2[3] - y1

    # normalized mask in union bbox
    bw = x2 - x1
    bh = y2 - y1

    mask1 = get_normal_mask(bw, bh, [x1_1, y1_1, x2_1, y2_1], nw, nh)
    mask2 = get_normal_mask(bw, bh, [x1_2, y1_2, x2_2, y2_2], nw, nh)
    
    # convert the masks to dual_mask = [nh, nw, 2], mask1 = dual_mask[:,:,0], mask2 = dual_mask[:,:,1]
    dual_mask = np.stack([mask1, mask2], axis = -1)
       
    return dual_mask
