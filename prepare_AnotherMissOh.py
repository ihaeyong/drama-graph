import numpy as np
import cv2

import json
import os

if __name__ == '__main__':

    # Load shot images and bbox from json

    # Set path AnotherMissOh Json
    json_dir = './data/AnotherMissOh/AnotherMissOh_Visual/AnotherMissOh01_visual.json'


    with open(json_dir, encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # Set path AnotherMissOh visual data
    img_path = './data/AnotherMissOh/AnotherMissOh_images/AnotherMissOh01/'
    save_path = './data/AnotherMissOh/Face/imgs/'

    count = 1

    for i in range(len(json_data['visual_results'])):
        for j in range(len(json_data['visual_results'][i]['image_info'])):

            try:
                id_name = json_data['visual_results'][i]['image_info'][j]['persons'][0]['person_id']
                face_bbox = json_data['visual_results'][i]['image_info'][j]['persons'][0]['person_info'][
                    'face_rect']
                full_bbox = json_data['visual_results'][i]['image_info'][j]['persons'][0]['person_info'][
                    'full_rect']

                img = cv2.imread(img_path + json_data['visual_results'][i]['vid'][-9:].replace('_', '/') + '/' +
                                 json_data['visual_results'][i]['image_info'][j]['frame_id'][-16:] + '.jpg')

                face_crop = img[face_bbox['min_y']:face_bbox['max_y'], face_bbox['min_x']:face_bbox['max_x']]
                face_resize = cv2.resize(face_crop, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)

                if not (os.path.isdir(save_path + id_name)):
                    os.makedirs(os.path.join(save_path + id_name))

                    cv2.imwrite(save_path + id_name + '/' + str(count) + '.jpg', face_resize)
                else:
                    cv2.imwrite(save_path + id_name + '/' + str(count) + '.jpg', face_resize)

                count = count + 1

            except:
                id_name = []
