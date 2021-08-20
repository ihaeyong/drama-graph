# Git clone this repository.
```
>> git clone https://github.com/ihaeyong/drama-graph.git
```

# Prepare env. and install requirements:
```
1. create conda env.
>> conda create -n vtt_env python=3.6
>> source activate vtt_env
2. install pytorch and lib. 
>> conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
>> pip install -r requirements.txt
```

# Download AnotherMissOh datasets

Unzip the anothermissoh dataset,

```
>> mkdir ./data
>> cd data
>> unzip datasets.zip
```

You can set your data path to 'Yolo_v2_pytorch/src/anotherMissOh_dataset.py' as follows:

```
img_path = './data/AnotherMissOh/AnotherMissOh_images/AnotherMissOh01/'
json_dir = './data/AnotherMissOh/AnotherMissOh_Visual/AnotherMissOh01_visual.json'
```


# Drama-graph model

We mainly use [YOLOv2-pytorch](https://github.com/uvipen/Yolo-v2-pytorch). 

You could find all trained models in [this link](https://drive.google.com/drive/folders/185sGBHO8v4SAVPaMnNJzDF8AOMhILjwM?usp=sharing) and make 'pre_model' folder and put the models. 

We finetuned YOLOv2 w.r.t 20 persons for about 50 epoches as follows:

### Train model:
train the integrated model from scratch. 

For place recognition, make 'pre_model' folder and put [places365 pre-trained model](https://drive.google.com/file/d/1fe-CnmM-1XcGBCPxtF3L4vjM7s0OJA6-/view?usp=sharing).

```
>> ./scripts/train_models.sh #gpu
```

train sound event model from scratch

```
>> ./scripts/train_sound_event.sh #gpu
```
The trained model is saved in `./sound_event_detection/checkpoint/torch_model.pt`


### Test model:
test the integrated model on testset
```
>> ./scripts/eval_models.sh #gpu
```


#### Evaluation
mAP for person
```
>> python eval_mAP.py -rtype person
```

mAP for behavior
```
>> python eval_mAP.py -rtype behave
```

mAP for face
```
>> python eval_mAP.py -rtype face
```

mAP for relation
```
>> python eval_mAP.py -rtype relation
```

mAP for object
```
>> python eval_mAP.py -rtype object
```

accuracy for sound event and its visualization.
sed_vis folder should be in the directory from which you run the file(`./`), 
so here the directory is (`drama-graph/sed_vis/`).
```
>> ./scripts/eval_sound_event.sh
>> ./scripts/inference_sound_event.sh

```
#### Performances
| model            | trainset | validation | test |
|------------------|----------|------------|------|
| person detection |   54.2%  |    50.6%   | 47.3%|
| face detection   |   43.9%  |    25.83   | 26.6%|
| emotion          |  72.6%   |    80.6%   | 66.9%|
| behavior         |  17.43%  |    3.9%    | 4.89%|
| object detection |  2.18%   |    1.17%   | 1.33%|
| predicate        |  88.1%   |    88.8%   | 85.9%|
| place            |  95.3%   |    68.2%   | 63.4%|
| sound event      |  89.6%   |    69.0%   | 62.5%|

#### Acknowledgements

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2017-0-01780, The technology development for event recognition/relational reasoning and learning knowledge based system for video understanding)
