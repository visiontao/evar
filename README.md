# Explainable Video Action Reasoning via Prior Knowledge and State Transitions
Tao Zhuo, Zhiyong Cheng, Peng Zhang, Yongkang Wong, Mohan Kankanhalli

Our paper can be found here: https://arxiv.org/abs/1908.10700 \
CAD-120 dataset can be found here: http://pr.cs.cornell.edu/humanactivities/data.php

# Setup
Ubuntu 16.04 \
Keras  \
Python2.7 

# Annotations
## 1. video_clips.txt: 
e.g. "Subject5, taking_medicine, 0126143431, 70, 135, open", it denotes follows:

**Subject5:** person id \
**taking_medicine:** video label \
**0126143431:** video id \
**70:** starting frame \
**135:** ending frame \
**open:** action  

## 2. splits: spliting videos for training and testing
## 3. knowledge: concerned objects, attributes, relationships and actions 
## 4. states: attributes and relationships for training.
## 5. all: annotated objects, attributes, relationships for all videos

**An example of dataloader (json file):**

    file_name_json = os.path.join('annotations/states/attr_01.json')    
    with open(file_name_json, 'r') as f:
        data_anno = json.load(f)

    for k in range(len(data_anno)):
        person_id = data_anno[k]['person_id']
        video_label = data_anno[k]['video_label']
        video_id = data_anno[k]['video_id']
        obj_label = data_anno[k]['obj_label']
        frame_id = data_anno[k]['frame_id']
        roi = data_anno[k]['roi']
        attr_label = data_anno[k]['attr_label']

# Citation
If our code and annotations are useful for you, please cite the following paper:

@article{zhuo2019explainable,
  title={Explainable Video Action Reasoning via Prior Knowledge and State Transitions},
  author={Zhuo, Tao and Cheng, Zhiyong and Zhang, Peng and Wong, Yongkang and Kankanhalli, Mohan},
  journal={ACM Multimedia},
  year={2019}
}

# Contact
Tao Zhuo (zhuotao@nus.edu.sg)
