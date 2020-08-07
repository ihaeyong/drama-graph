import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import glob
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import json
import pdb

# define person classes
PersonCLS = ['Dokyung', 'Haeyoung1', 'Haeyoung2', 'Sukyung', 'Jinsang',
            'Taejin', 'Hun', 'Jiya', 'Kyungsu', 'Deogi',
            'Heeran', 'Jeongsuk', 'Anna', 'Hoijang', 'Soontack',
            'Sungjin', 'Gitae', 'Sangseok', 'Yijoon', 'Seohee', 'none']

# define behavior
PBeHavCLS = ["stand up","sit down","walk","hold","hug",
             "look at/back on",
             "drink","eat",
             "point out","dance", "look for","watch",
             "push away", "wave hands",
             "cook", "sing", "play instruments",
             "call", "destroy",
             "put arms around each other's shoulder",
             "open", "shake hands", "wave hands",
             "kiss", "high-five", "write", "none"]

# define person to person relations
P2PRelCLS = ['Friendly', 'Unfriendly', 'none']

# define object classes 

# ObjectCLS = ["apron", "tie", "bottle", "table", "chair(stool)", "vase", "couch", "remote_control", "potted_plant", 
#             "glass", "handbag", "bag", "lamp", "food", "fork", "basket", "clock", "computer", "TV", "paper(report)", "glasses", 
#             "binder", "can/beer", "plate", "phone", "flower_pot", "book", "frame", "car", "bed", "tree", "flower", "watch", 
#             "wine_glass", "can", "suitcase", "bookshelf", "shoes", "box", "door", "cabinet", "desk", "umbrella", "laptop", "bowl", 
#             "microphone", "fan", "refrigerator", "shelf", "beer_bottle", "apple", "pillow", "sofa", "window", "water_bottle", 
#             "coffee_maker", "keyboard", "plant", "tooth_brush", "machine", "bicycle", "cup_or_mug", "clothes", "cap", "bow_tie", 
#             "truck", "traffic_light", "stove", "chopsticks", "spoon", "pot", "sink", "racket", "dog", "horse", "remote", 
#             "frying_pan", "purse", "towel", "plastic_bag", "hat", "bus", "microwave", "soccer_ball", "bench", "notebook", 
#             "cigarette", "cakes", "backpack", "teddy_bear", "photo", "wine_bottle", "butterfly", "drawer", "sunglasses", "hammer", 
#             "knife", "sandwitch", "carrot", "cushion", "tape_player", "sports_ball", "envelope", "washingmachine", "blender", 
#             "pencil_box", "dish", "paper_towel", "lipstick", "ladle", "paper_(report)", "camera", "baseball_bat", "kettle", "blanket", 
#             "motorcycle", "candle", "filing_cabinet", "digital_clock", "brick", "bird", "coffeepot", "trumpet", "cat", "grass", "card", 
#             "detergent_box", "board", "scissors", "star", "mouse", "broccoli", "building", "bat", "train", "nail", 
#             "pencil_sharpener", "airplane", "poster", "guitar", "floor", "plate_rack", "accordion", "swine", "spatula", "rabbit", 
#             "oven", "saxophone", "tennis_ball", "cow", "banana", "stop_sign", "boat", "cream", "tennis_racket", "orange", "baseball", 
#             "skateboard", "perfume", "elephant", "axe", "hamburger"]

# this is what we decided to use because there is too many. 47 classes
ObjectCLS = ["bag", "bed", "beer_bottle", "bicycle", "book", "bookshelf", "bottle", "bowl", "box", "bus", "can_beer", 
            "car", "chair(stool)", "computer", "couch", "desk", "door", "flower", "food", "frame", "glass", "glasses", 
            "handbag", "hat", "lamp", "laptop", "microphone", "paper(report)", "phone", "plant", "plate", "pot", 
            "potted_plant", "refrigerator", "shoes", "sofa", "spoon", "table", "tie", "traffic_light", "tree", "TV", 
            "umbrella", "vase", "watch", "window", "wine_glass"]


P2ORelCLS = ['none', 'wearing', 'on', 'with', 'in front of', 'has', 'in', 'near', 'attached to', 'under', 'at', 'N_P', 'N_O']

# define face classes
FaceCLS = ['Dokyung', 'Haeyoung1', 'Haeyoung2', 'Sukyung', 'Jinsang',
            'Taejin', 'Hun', 'Jiya', 'Kyungsu', 'Deogi',
            'Heeran', 'Jeongsuk', 'Anna', 'Hoijang', 'Soontack',
            'Sungjin', 'Gitae', 'Sangseok', 'Yijoon', 'Seohee', 'none']

def Splits(num_episodes):
    '''
    split the total number of episodes into three : train, val, test
    '''
    train = [*range(1, 6), *range(9,num_episodes)]
    val = [] #num_episodes-3
    #test = range(num_episodes-2, num_episodes)
    test = [7,8]

    return train, val, test

def SortFullRect(image,label, is_train=True):

    width, height = (1024, 768)
    width_ratio = 448 / width
    height_ratio = 448 / height

    fullrect_list = []
    fullbehav_list = []
    fullobj_list = []
    fullrelation_list = []
    facerect_list = []
    num_batch = len(label[0]) # per 1 video clip

    # set sequence length
    s_frm = 0
    e_frm = num_batch

    max_batch = 10
    if num_batch > max_batch and is_train:
        s_frm = np.random.choice(num_batch-max_batch, 1)[0]
        e_frm = s_frm + max_batch
    elif is_train is False:
        s_frm = 0
        e_frm = max_batch

    image_list = []
    frame_id_list = []
    for frm in range(s_frm, e_frm):
        try:
            label_list = []
            behavior_list = []
            face_list = []

            frame_id = label[0][frm]['frame_id']
            frame_id_list.append(frame_id)
            for p, p_id in enumerate(label[0][frm]['persons']['person_id']):


                p_label = PersonCLS.index(p_id)
                if p_label > 20:
                    print("sort full rect index error{}".format(p_label))

                full_rect = label[0][frm]['persons']['full_rect'][p]

                # behavior label
                behavior = label[0][frm]['persons']['behavior'][p]
                behavior_label = PBeHavCLS.index(behavior)

                # face label
                face_label = FaceCLS.index(p_id)
                if face_label > 20:
                    print("sort face rect index error{}".format(face_label))

                face_rect = label[0][frm]['persons']['face_rect'][p]

                #scale:
                xmin = max(full_rect[0] * width_ratio, 0)
                ymin = max(full_rect[1] * height_ratio, 0)
                xmax = min((full_rect[2]) * width_ratio, 448)
                ymax = min((full_rect[3]) * height_ratio, 448)
                full_rect = [xmin,ymin,xmax,ymax]

                # face rcet scale
                xmin = max(face_rect[0] * width_ratio, 0)
                ymin = max(face_rect[1] * height_ratio, 0)
                xmax = min((face_rect[2]) * width_ratio, 448)
                ymax = min((face_rect[3]) * height_ratio, 448)
                face_rect = [xmin, ymin, xmax, ymax]

                temp_label = np.concatenate((full_rect, [p_label]), 0)
                label_list.append(temp_label)
                behavior_list.append(behavior_label)

                face_temp_label = np.concatenate((face_rect, [face_label]), 0)
                face_list.append(face_temp_label)
        except:
            continue

    for frm in range(s_frm, e_frm):
        try:
            object_list = []
            # relation_list = []
            for p, p_id in enumerate(label[0][frm]['objects']['object_id']):
                # we need to account for 'person' and remove person from the list
                if p_id == 'person':
                    continue
                p_label = ObjectCLS.index(p_id)
                r_label = P2ORelCLS.index(label[0][frm]['objects']['relation'][p])
                if p_label > 47:
                    print("sort full rect index error{}".format(p_label))
                # object labels
                object_rect = label[0][frm]['objects']['object_rect'][p]

                #scale:
                xmin = max(object_rect[0] * width_ratio, 0)
                ymin = max(object_rect[1] * height_ratio, 0)
                xmax = min((object_rect[2]) * width_ratio, 448)
                ymax = min((object_rect[3]) * height_ratio, 448)
                object_rect = [xmin,ymin,xmax,ymax]

                temp_label = np.concatenate((object_rect, [p_label], [r_label]), 0)
                # temp_label = np.concatenate((object_rect, [p_label]), 0)
                # relation_list.append(r_label)
                object_list.append(temp_label)
        except:
            continue

        if len(label_list) > 0 and is_train:
            fullrect_list.append(label_list)
            fullbehav_list.append(behavior_list)
            image_list.append(image[frm])
            fullobj_list.append(object_list)
            # fullrelation_list.append(relation_list)
            facerect_list.append(face_list)

        else: # for test
            fullrect_list.append(label_list)
            fullbehav_list.append(behavior_list)
            image_list.append(image[frm])
            fullobj_list.append(object_list)
            # fullrelation_list.append(relation_list)
            facerect_list.append(face_list)


    if is_train:
        return image_list, fullrect_list, fullbehav_list, fullobj_list, facerect_list
    return image_list, fullrect_list, fullbehav_list, fullobj_list, facerect_list, frame_id_list

class AnotherMissOh(Dataset):
    def __init__(self, dataset, img_path, json_path, display_log=True):

        self.display_log = display_log
        self.init_clips(img_path)
        self.load_json(dataset, img_path, json_path)

    def init_clips(self, img_path):
        self.cnt_clips = 0
        self.img_path = img_path

        self.img_size = (1024, 768)
        self.img_scaled_size = (448, 448)

        tform = [
            Resize(self.img_scaled_size),  # should match to Yolo_V2
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # should match to Yolo_V2
        ]
        self.transformations = Compose(tform)

        '''
        clips = {
            'episode' : [],
            'clip' : [],
            'start_time' : [],
            'end_time' : [],
            'vid' : [],
            'img_size' : [],
            'img_scaled_size' : [],
            'image_info' : []
        }

        image_info = {
            'frame_id': [],
            'place' : [],
            'persons' : [] # self.person
        }

        persons = {
            'person_id': [],
            'face_rect' : [],
            'full_rect' : [],
            'behavior' : [],
            'predicate' : [],
            'emotion' : [],
            'face_rect_score' : [],
            'full_rect_score' : []
        }
        '''

    def load_json(self, dataset, img_path, json_path):

        self.clips = []

        for episode in dataset:
            img_dir = img_path + 'AnotherMissOh{:02}/'.format(episode)
            # json_dir = json_path + 'AnotherMissOh{:02}_ver3.2.json'.format(episode)
            json_dir = json_path + 'AnotherMissOh{:02}_visual.json'.format(episode)
            if self.display_log:
                print('imag_dir:{}'.format(img_dir))
                print('json_dir:{}'.format(json_dir))

            with open(json_dir, encoding='utf-8') as json_file:
                json_data = json.load(json_file)

            for i in range(len(json_data['visual_results'])):
                clip = {}
                clip['episode'] = []
                clip['clip'] = []
                clip['start_time'] = []
                clip['end_time'] = []
                clip['vid'] = []
                clip['image_info'] = []

                if self.display_log:
                    print(
                        "***{}th episode***{}th clips**********".format(episode, i))
                    print("['visual_results'][{}]['start_time']:{}".format(
                        i, json_data['visual_results'][i]['start_time']))
                    print("['visual_results'][{}]['end_time']:{}".format(
                        i, json_data['visual_results'][i]['end_time']))
                    print("['visual_results'][{}]['vid']:{}".format(
                        i, json_data['visual_results'][i]['vid'].replace('_', '/')))
                    print("['visual_results'][{}]['img_size']:{}".format(
                        i, img_size))
                    print("['visual_results'][{}]['img_scaled_size']:{}".format(
                        i, img_scaled_size))
                    print("['visual_results'][{}]['episode']:{}".format(i, episode))

                clip['episode'].append(episode)
                clip['clip'].append(i)
                clip['start_time'].append(json_data['visual_results'][i]['start_time'])
                clip['end_time'].append(json_data['visual_results'][i]['end_time'])
                clip['vid'].append(json_data['visual_results'][i]['vid'].replace('_', '/'))

                for j, info in enumerate(json_data['visual_results'][i]['image_info']):
                    image_info = {}
                    image_info['frame_id'] = []
                    image_info['place'] = []
                    image_info['objects'] = {}
                    image_info['persons'] = {}

                    if self.display_log:
                        print("=============={}th frame==========".format(j))

                    img_file = img_dir + json_data['visual_results'][i]['vid'].replace('_', '/')[-8:] + '/' + info['frame_id'][-16:] + '.jpg'
                    image_info['frame_id'].append(img_file)
                    image_info['place'].append(info['place'])

                    # add for the objects
                    image_info['objects']['object_id']=[]
                    image_info['objects']['object_rect']=[]
                    image_info['objects']['relation']=[]
                    for k, obj in enumerate(info['objects']):
                        if obj['object_id'] in ObjectCLS:
                            image_info['objects']['object_id'].append(obj['object_id'])
                            object_bbox = obj['object_rect']
                            if (object_bbox['min_y'] == "" 
                                or object_bbox['max_y'] == "" 
                                or object_bbox['min_x'] == "" 
                                or object_bbox['max_x'] == ""):
                                object_rect = []
                                continue
                            else:
                                object_rect = [object_bbox['min_x'], object_bbox['min_y'], 
                                               object_bbox['max_x'], object_bbox['max_y']]
                        image_info['objects']['object_rect'].append(object_rect)
                        image_info['objects']['relation'].append('N_P')
                    # objects

                    image_info['persons']['person_id'] = []
                    image_info['persons']['face_rect'] = []
                    image_info['persons']['full_rect'] = []
                    image_info['persons']['behavior'] = []
                    image_info['persons']['predicate'] = []
                    image_info['persons']['emotion'] = []
                    image_info['persons']['face_rect_score'] = []
                    image_info['persons']['full_rect_score'] = []

                    image_info['persons']['related_object_id']=[]
                    image_info['persons']['related_object_rect']=[]
                    for k, person in enumerate(info['persons']):
                        if self.display_log:
                            print("-------{}th person-----------".format(k))

                        image_info['persons']['person_id'].append(person['person_id'])
                        
                        #import pdb; pdb.set_trace()
                        for j, robj in enumerate(person['related_objects']):
                            image_info['persons']['related_object_id'].append(robj['related_object_id'])
                            image_info['objects']['object_id'].append(robj['related_object_id'])
                            robj_bbox = robj['related_object_rect']
                            if (robj_bbox['min_y'] == "" 
                                or robj_bbox['max_y'] == "" 
                                or robj_bbox['min_x'] == "" 
                                or robj_bbox['max_x'] == ""):
                                related_object_rect = []
                                continue
                            else:
                                related_object_rect = [robj_bbox['min_x'], robj_bbox['min_y'], 
                                                       robj_bbox['max_x'], robj_bbox['max_y']]
                            image_info['persons']['related_object_rect'].append(related_object_rect)
                            image_info['objects']['object_rect'].append(related_object_rect)
                            if j > 0:
                                image_info['objects']['relation'].append('none')
                            else:
                                image_info['objects']['relation'].append(person['person_info']['predicate'])

                        face_bbox = person['person_info']['face_rect']
                        if (face_bbox['min_y'] == "" 
                            or face_bbox['max_y'] == "" 
                            or face_bbox['min_x'] == "" 
                            or face_bbox['max_x'] == ""):
                            face_rect = []
                            continue
                        else:
                            face_rect = [face_bbox['min_x'], face_bbox['min_y'], face_bbox['max_x'], face_bbox['max_y']]
                        image_info['persons']['face_rect'].append(face_rect)
                        full_bbox = person['person_info']['full_rect']
                        if (full_bbox['min_y'] == "" 
                            or full_bbox['max_y'] == "" 
                            or full_bbox['min_x'] == "" 
                            or full_bbox['max_x'] == ""):
                            full_rect = []
                            continue
                        else:
                            full_rect = [full_bbox['min_x'], full_bbox['min_y'], full_bbox['max_x'], full_bbox['max_y']]
                        image_info['persons']['full_rect'].append(full_rect)
                        image_info['persons']['behavior'].append(person['person_info']['behavior'])
                        image_info['persons']['predicate'].append(person['person_info']['predicate'])
                        image_info['persons']['emotion'].append(person['person_info']['emotion'])
                        image_info['persons']['face_rect_score'].append(person['person_info']['face_rect_score'])
                        image_info['persons']['full_rect_score'].append(person['person_info']['full_rect_score'])

                    clip['image_info'].append(image_info)

                self.clips.append(clip)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, item):
        image_info = self.clips[item]['image_info']

        img_list = []
        for it, frame in enumerate(image_info):
            img = Image.open(frame['frame_id'][0]).convert('RGB')
            img = self.transformations(img)
            img_list.append(img)

        return img_list, image_info
