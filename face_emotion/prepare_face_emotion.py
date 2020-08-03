#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys,os,time
import numpy as np
import json

db_path = '/home/jhchoi/datasets4/VTT_AMO/db/'


# In[2]:


def emo_char_idx(emo):
    # 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
    if emo == 'angry' or emo == 'anger':
        return 0
    elif emo == 'disgust':
        return 1
    elif emo == 'fear':
        return 2
    elif emo == 'happy' or emo == 'happiness':
        return 3
    elif emo == 'sad' or emo == 'sadness':
        return 4
    elif emo == 'surprise':
        return 5
    elif emo == 'neutral':
        return 6
    else:
        print('error, '+emo)


# In[3]:


with open(os.path.join(db_path,'AnotherMissOh_Visual_full.json')) as f:
    anno_full = json.load(f)


# In[4]:


emo_dict = dict()
for k in sorted(list(anno_full.keys())):
    epi, ann, idx = k.split('_')
    
    if epi not in emo_dict.keys():
        emo_dict[epi] = dict()
    if ann not in emo_dict[epi].keys():
        emo_dict[epi][ann] = dict()
    
    emo_dict[epi][ann][idx] = list()
    
    for a in anno_full[k]:
        for p in a['persons']:
            pi = p['person_info']
            p_emo = {'emotion': emo_char_idx(pi['emotion'].lower()),
                    'face_rect': [pi['face_rect']['min_x'], pi['face_rect']['min_y'], pi['face_rect']['max_x'], pi['face_rect']['max_y']],
                     'img': a['frame_id'].split('_')[-1]
                    }
            emo_dict[epi][ann][idx].append(p_emo)
        
    
    if len(emo_dict[epi][ann][idx]) == 0:
        emo_dict[epi][ann].pop(idx,None)
        
    if len(emo_dict[epi][ann].keys()) == 0:
        emo_dict[epi].pop(ann,None)
                


# In[6]:


with open(os.path.join(db_path,'AnotherMissOh_Visual_emo.json'), 'w') as f:
    json.dump(emo_dict, f)


# In[ ]:




