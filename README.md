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

You could find all trained models in [this link](https://drive.google.com/drive/folders/1LvDpPkkZ_18Zhf70rXUDaLoGFp2x6M5G) and make 'pre_model' folder and put the models. 

We finetuned YOLOv2 w.r.t 20 persons for about 50 epoches as follows:

### Train model:
train model from scratch

```
>> ./scripts/train_main.py #gpu
```
[trained model](https://drive.google.com/drive/folders/185sGBHO8v4SAVPaMnNJzDF8AOMhILjwM?usp=sharing)

### Test model:
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


#### Acknowledgements

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2017-0-01780, The technology development for event recognition/relational reasoning and learning knowledge based system for video understanding)
