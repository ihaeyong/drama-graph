## SWRC

### Install pakages
```
> pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz --no-deps
> apt-get install default-jdk
````

Downloads [stanford-corenlp-4.0.0.zip](https://stanfordnlp.github.io/CoreNLP/history.html) and unzip the downloaded package in data folder.


### Execute
```
> ./train_swrc.sh
```

### Output

`data/output/graph.json`

- dictionary
  - key: scene id
  - value: scene graph
- scene graph
  - char_background
    - scene 내에 등장하는 등장인물의 배경지식.
    - form: subject - relation - object
  - common_sense
    - scene 내에 등장하는 Noun의 ConceptNet 지식.
    - form: subject - relation - object
  - entity_background
    - scene 내에 등장하는 Noun의 wiki 기반 지식. (from Acryl)
    - form: subject - relation - object
  - triples
    - open IE로 추출한 triple events.
    - form: subject - relation - object
  - frames
    - frameBERT로 추출한 frame events.
    - form: frame - lu - args
      - frame: 추출된 frame type
      - lu: 추출된 frame의 trigger.
      - args: frame의 arguments

`data/output/graphs`

- graph visualization
  - triples & char_background로 이루어진 그래프.
  - frames로 이루어진 그래프.
