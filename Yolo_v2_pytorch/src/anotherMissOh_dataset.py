import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import glob
from torchvision.transforms import Compose, Resize
from PIL import Image
import json

class MissOhDataset(Dataset):
    def __init__(self, image_size=448):
        img_path = './data/AnotherMissOh/AnotherMissOh_images/AnotherMissOh01/'
        json_dir = './data/AnotherMissOh/AnotherMissOh_Visual/AnotherMissOh01_visual.json'

        with open(json_dir, encoding='utf-8') as json_file:
            json_data = json.load(json_file)

        self.img_list = []
        self.anno_list = []

        for i in range(len(json_data['visual_results'])):
            for j in range(len(json_data['visual_results'][i]['image_info'])):
                label = []
                for k in range(len(json_data['visual_results'][i]['image_info'][j]['persons'])):
                    try:
                        id_name = json_data['visual_results'][i]['image_info'][j]['persons'][k]['person_id']
                        full_bbox = json_data['visual_results'][i]['image_info'][j]['persons'][k]['person_info'][
                            'full_rect']
                        if full_bbox['min_y'] == "" or full_bbox['max_y'] == "" or full_bbox['min_x'] == "" or full_bbox['max_x'] == "":
                            continue
                        else:
                            temp_label = [full_bbox['min_x'], full_bbox['min_y'], full_bbox['max_x'], full_bbox['max_y'], 0]
                            label.append(temp_label)
                    except:
                        id_name = []

                self.img_list.append(img_path + json_data['visual_results'][i]['vid'][-9:].replace('_', '/') + '/' +
                                     json_data['visual_results'][i]['image_info'][j]['frame_id'][-16:] + '.jpg')
                self.anno_list.append(label)


        self.image_size = (image_size, image_size)

        self.num_images = len(self.img_list)


    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        image_path = os.path.join(self.img_list[item])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformations = Compose([Resize(self.image_size)])

        objects = self.anno_list[item]
        #
        # image, objects = transformations((image, objects))
        image = transformations(Image.fromarray(image))

        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)

class MissOhDatasetTest(Dataset):
    def __init__(self, image_size=448):
        img_path = './data/AnotherMissOh/AnotherMissOh_images/AnotherMissOh01/'
        json_dir = './data/AnotherMissOh/AnotherMissOh_Visual/AnotherMissOh01_visual.json'

        with open(json_dir) as json_file:
            json_data = json.load(json_file)

        self.img_list = []
        self.anno_list = []

        for i in range(len(json_data['visual_results'])):
            for j in range(len(json_data['visual_results'][i]['image_info'])):
                label = []
                for k in range(len(json_data['visual_results'][i]['image_info'][j]['persons'])):
                    try:
                        id_name = json_data['visual_results'][i]['image_info'][j]['persons'][k]['person_id']
                        full_bbox = json_data['visual_results'][i]['image_info'][j]['persons'][k]['person_info'][
                            'full_rect']
                        if full_bbox['min_y'] == "" or full_bbox['max_y'] == "" or full_bbox['min_x'] == "" or full_bbox['max_x'] == "":
                            continue
                        else:
                            temp_label = [full_bbox['min_x'], full_bbox['min_y'], full_bbox['max_x'], full_bbox['max_y'], 0]
                            label.append(temp_label)
                    except:
                        id_name = []

                self.img_list.append(img_path + json_data['visual_results'][i]['vid'][-9:].replace('_', '/') + '/' +
                                     json_data['visual_results'][i]['image_info'][j]['frame_id'][-16:] + '.jpg')
                self.anno_list.append(label)


        self.image_size = image_size

        self.num_images = len(self.img_list)


    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        image_path = os.path.join(self.img_list[item])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformations = Compose([Resize(self.image_size)])

        objects = self.anno_list[item]

        image, objects = transformations((image, objects))

        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)
