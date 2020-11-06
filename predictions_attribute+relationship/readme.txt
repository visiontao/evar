This folder contains both the attribute and relationship predictions.

1. Since most of the recent methods only outputs an action label for a video sequence, for the purpose of comparison, we split a long video sequence into several clips that involve a single action only, please the annotations/video_clips.txt. Besides, for the purpose of efficient processing, some frames are removed for the original video and there is no performed actions druing those frames.

2. The number in file ``1_30_attr.json'' and ``1_30_rel.json'' denotes the start and end frame of the video, please see the annotations/video_clips.txt

3. For both attribute and relationship predictions, the value ``-1'' denotes the objects are occluded or out of view, and thus there is no prediction on that frame.
   For attribute resutls, the prediction is a 2-dim vector, corresponding to annotations/knowledge/attribute_list.txt
   For relation results, the prediction is a 6-dim vector, corresponding to annotations/knowledge/relation_list.txt
