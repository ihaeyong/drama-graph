# Git clone this repository.
```
>> git clone https://github.com/ihaeyong/drama-graph.git
```

# Prepare virtual environments and install requirements:
```
1. create conda env.
>> conda create -n vtt_env python=3.6
>> source activate vtt_env
2. install pytorch and lib. 
>> conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
>> pip install -r requirements.txt
```

# Download datasets

We used AnotherMissOh dataset.

Currently, only 1 episode of AnotherMissOh data was used for learning, but it will be added other episodes continuously.

All episodes have image files and annotated json file.

You could find AnotherMissOh dataset in [this link](https://drive.google.com/open?id=1jcAhHCmq3fyhJ9Ggm9EA1Tf_xT3Roe48). 

Password is required.

After unzip,

```
>> mkdir ./data
>> cd data
>> unzip datasets.zip
```

you set your data path in 'Yolo_v2_pytorch/src/anotherMissOh_dataset.py' as below

```
img_path = './data/AnotherMissOh/AnotherMissOh_images/AnotherMissOh01/'
json_dir = './data/AnotherMissOh/AnotherMissOh_Visual/AnotherMissOh01_visual.json'
```

# Face Recognition 
We mainly use ArcFace: Additive Angular Margin Loss for Deep Face Recognition method. The code is based on [ArcFace-pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) implementation.

This repository is not completed. We will update new training code and add Person re-identification for undetected face.

<img src="https://user-images.githubusercontent.com/37200420/77849986-86800e00-720a-11ea-82e7-59111a8963f4.JPG">
<img src="https://user-images.githubusercontent.com/37200420/77849994-8e3fb280-720a-11ea-9dde-c18e51001996.JPG" width="40%">
<img src="https://user-images.githubusercontent.com/37200420/77849995-90a20c80-720a-11ea-9d64-b567e2b56778.JPG" width="40%">

### Train face :
```
1. prepare AnotherMissOh dataset 
>> ./script/prepare_face_image.sh 
after set your json and visual data path
2. run train.py
>> ./script/train_face.sh

```

### Trained face models
You could find all trained models.

And make 'pre_model' folder and put the models.

# Person Detection
We mainly use YOLOv2.
The code is based on [YOLOv2-pytorch](https://github.com/uvipen/Yolo-v2-pytorch) implementation.

We are constantly training about person detection.

### Train person detection:
```
1. set visdom for learning curves
>> python -m visdom.server -port 6005
2. train person detection  
>> ./scripts/train_main.py #gpu

```

### Trained person detection models
You could find all trained models in [this link](https://drive.google.com/drive/folders/1LvDpPkkZ_18Zhf70rXUDaLoGFp2x6M5G)

And make 'pre_model' folder and put the models.


# Person identification using face and body

### Train person identification:
```
1. set visdom for learning curves
>> python -m visdom.server -port 6005
2. train person identification  
>> ./scripts/train_main.py #gpu

```

# Demo:
### person face detection
```
>> ./scripts/eval_face.sh #gpu

```
### person detection
```
>> ./scripts/eval_models.sh #gpu

```
### person identification using face
```
>> ./scripts/eval_models.sh #gpu

```

# To Do List:
```
1. visualization bbox for person detection
2. return bbox
3. change some functions to latest pytorch function
```
