import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import glob
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import json

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

# define face classes
FaceCLS = ['Dokyung', 'Haeyoung1', 'Haeyoung2', 'Sukyung', 'Jinsang',
            'Taejin', 'Hun', 'Jiya', 'Kyungsu', 'Deogi',
            'Heeran', 'Jeongsuk', 'Anna', 'Hoijang', 'Soontack',
            'Sungjin', 'Gitae', 'Sangseok', 'Yijoon', 'Seohee', 'none']

# define person to person relations
P2PRelCLS = ['Friendly', 'Unfriendly', 'none']

# define object classes
ObjectCLS = ['Phone', 'Stick', 'none']
P2ORelCLS = ['Holding', 'Wearing', 'none']

def Splits(num_episodes):
    '''
    split the total number of episodes into three : train, val, test
    '''
    train = range(1, num_episodes-3)
    val = num_episodes-3
    test = range(num_episodes-2, num_episodes)

    return [*train], [val], [*test]

def SortFullRect(image, label, is_train=True):

    width, height = (1024, 768)
    width_ratio = 448 / width
    height_ratio = 448 / height

    fullrect_list = []
    fullbehav_list = []
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

            for p, p_id in enumerate(label[0][frm]['persons']['person_id']):

                frame_id = label[0][frm]['frame_id']
                frame_id_list.append(frame_id)

                p_label = PersonCLS.index(p_id)
                face_label = FaceCLS.index(p_id)

                if p_label > 20:
                    print("sort full rect index error{}".format(p_label))

                if face_label > 20:
                    print("sort face rect index error{}".format(face_label))

                full_rect = label[0][frm]['persons']['full_rect'][p]

                # behavior label
                behavior = label[0][frm]['persons']['behavior'][p]
                behavior_label = PBeHavCLS.index(behavior)

                # face rect
                face_rect = label[0][frm]['persons']['face_rect'][p]

                # scale:
                xmin = max(full_rect[0] * width_ratio, 0)
                ymin = max(full_rect[1] * height_ratio, 0)
                xmax = min((full_rect[2]) * width_ratio, 448)
                ymax = min((full_rect[3]) * height_ratio, 448)
                full_rect = [xmin, ymin, xmax, ymax]

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

        if len(label_list) > 0 and is_train:
            fullrect_list.append(label_list)
            fullbehav_list.append(behavior_list)
            facerect_list.append(face_list)
            image_list.append(image[frm])
        else: # for test
            fullrect_list.append(label_list)
            fullbehav_list.append(behavior_list)
            facerect_list.append(face_list)
            image_list.append(image[frm])

    if is_train:
        return image_list, fullrect_list, fullbehav_list, facerect_list
    return image_list, fullrect_list, fullbehav_list, facerect_list, frame_id_list

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
            json_dir = json_path + \
                'AnotherMissOh{:02}_ver3.2.json'.format(episode)
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
                clip['start_time'].append(
                    json_data['visual_results'][i]['start_time'])
                clip['end_time'].append(
                    json_data['visual_results'][i]['end_time'])
                clip['vid'].append(json_data['visual_results']
                                   [i]['vid'].replace('_', '/'))

                num_persons = 0
                for j, info in enumerate(json_data['visual_results'][i]['image_info']):
                    image_info = {}
                    image_info['frame_id'] = []
                    image_info['place'] = []
                    image_info['persons'] = {}

                    if self.display_log:
                        print(
                            "=============={}th frame==========".format(j))

                    img_file = img_dir + json_data['visual_results'][i]['vid'].replace(
                        '_', '/')[-8:] + '/' + info['frame_id'][-16:] + '.jpg'
                    image_info['frame_id'].append(img_file)
                    image_info['place'].append(info['place'])

                    image_info['persons']['person_id'] = []
                    image_info['persons']['face_rect'] = []
                    image_info['persons']['full_rect'] = []
                    image_info['persons']['behavior'] = []
                    image_info['persons']['predicate'] = []
                    image_info['persons']['emotion'] = []
                    image_info['persons']['face_rect_score'] = []
                    image_info['persons']['full_rect_score'] = []
                    for k, person in enumerate(info['persons']):
                        if self.display_log:
                            print("-------{}th person-----------".format(k))

                        image_info['persons']['person_id'].append(
                            person['person_id'])
                        face_bbox = person['person_info']['face_rect']
                        if (face_bbox['min_y'] == ""
                            or face_bbox['max_y'] == ""
                            or face_bbox['min_x'] == ""
                                or face_bbox['max_x'] == ""):
                            face_rect = []
                            continue
                        else:
                            face_rect = [
                                face_bbox['min_x'], face_bbox['min_y'], face_bbox['max_x'], face_bbox['max_y']]
                        image_info['persons']['face_rect'].append(face_rect)
                        full_bbox = person['person_info']['full_rect']
                        if (full_bbox['min_y'] == ""
                            or full_bbox['max_y'] == ""
                            or full_bbox['min_x'] == ""
                                or full_bbox['max_x'] == ""):
                            full_rect = []
                            continue
                        else:
                            full_rect = [
                                full_bbox['min_x'], full_bbox['min_y'], full_bbox['max_x'], full_bbox['max_y']]
                            num_persons+=1
                        image_info['persons']['full_rect'].append(full_rect)
                        image_info['persons']['behavior'].append(
                            person['person_info']['behavior'])
                        image_info['persons']['predicate'].append(
                            person['person_info']['predicate'])
                        image_info['persons']['emotion'].append(
                            person['person_info']['emotion'])
                        image_info['persons']['face_rect_score'].append(
                            person['person_info']['face_rect_score'])
                        image_info['persons']['full_rect_score'].append(
                            person['person_info']['full_rect_score'])

                    clip['image_info'].append(image_info)

                if num_persons > 0:
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
