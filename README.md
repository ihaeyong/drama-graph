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


# Place model

You could find trained models in [this link](https://drive.google.com/file/d/1yqkTXGAGHqs_0B1Z5OsyGxL2sbfGbs6k/view?usp=sharing) and make 'checkpoint/clsf' folder and put the models. 

We finetuned sequantial architecture, Resnet50 pretrained with Places365 and lstm-based classifier with our own loss. 


### Train model:

```
>> ./scripts/train_main.py #gpu
```

### Test model:
```
>> ./scripts/eval_models.sh #gpu
```



#### Acknowledgements

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2017-0-01780, The technology development for event recognition/relational reasoning and learning knowledge based system for video understanding)
