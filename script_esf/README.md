# ESFWN-based ESL Annotator 
**ESFWN-based ESL Annotator**는 **E**vent **S**tructure **F**rame-annotated **W**ord**N**et-based **E**vent **S**tructure **L**exicon **Annnotator**의 약자로 문장을 입력으로 받아서 문장 내의 동사와 그 논항에 대해 "사건구조프레임"과 "의미역" 등 의미 정보를 출력하는 자동 주석기입니다. <br>
**ESFWN**은 영어 워드넷의 모든 동사에 대해 각 신셋마다 사건구조프레임 유형을 주석해 놓은 ESF-annotated WordNet이라는 어휘부 사전입니다.

## 주석기 구성요소 (Annotator Components)
1. 단어 중의성 해소 및 의미 주석 알고리즘 (ewiser & ewiser_wrapper)
2. 의미역 라벨링 도구 (AllenNLP SRL)
3. 사건구조프레임 주석 워드넷 (ESFWN)
4. 사건구조프레임 목록 (ESF_list)
5. 동사 불규칙 굴절 사전 (vInflection)

## 설치 (Installation)
1. [Anaconda3](https://www.anaconda.com/products/individual) 설치<br>

2. ewiser conda environment를 생성하고, 활성화<br>
   `conda create -n ewiser python=3.7 pip; conda activate ewiser`<br>

3. [pytorch-1.5.0](https://pytorch.org/get-started/locally/) 설치<br>

4. [torch_scatter, torch_sparse](https://github.com/rusty1s/pytorch_sparse) 설치<br>
   `pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.5.0+${CUDA}.html`<br>
   - ${CUDA}는 cpu, cu92, cu101, cu110 등 파이토치 설치 버전에 따라 선택.<br>
   - cpu만 사용하는 경우 torch_sparse 설치는 아래의 코드 실행<br>
     `pip install torch-sparse==0.6.7 -f https://pytorch-geometric.com/whl/torch-1.5.0+cpu.html`

5. [ewiser](https://github.com/SapienzaNLP/ewiser) 설치 <br>
    `git clone https://github.com/SapienzaNLP/ewiser.git`<br>
    `cd ewiser`<br>
    `pip install -r requirements.txt`<br>
    `pip install -e .`<br>

    - 만약 requirements의 각 툴이 설치가 안되면 각각 pip을 이용해 설치하면 됨.<br>
    - [fairseq](https://github.com/pytorch/fairseq)는 아래 코드로 설치.<br>
       `pip install fairseq==0.10.0`<br>
    - [nltk](https://nltk.org) 데이터 다운로드<br>
       `python`<br>
       `>>>import nltk`<br>
       `>>>nltk.download()`<br>
       화면에 오픈되는 테이블에서 Collections 탭에서 all 선택한 후 Download 버튼 클릭<br>

6. [spacy](https://spacy.io/) 설치<br>
   `pip install spacy`<br>
   `python -m spacy download en_core_web_sm`<br>

7. ewiser English checkpoints를 [여기](https://drive.google.com/file/d/11RyHBu4PwS3U2wOk-Le9Ziu8R3Hc0NXV/view)에서 다운로드해서 ewiser 폴더에 넣기.<br>

8. ewiser/bin/annotate.py 59번째줄 nlp = load(args.language.lower(), disable=['ner', 'parser'])에서 args.language.lower()대신 'en_core_web_sm' 삽입

9. ewiser 관련 툴 설치 확인<br>
   `python`<br>
   `import torch, torch_scatter, torch_sparse, numpy, nltk, h5py, joblib, fairseq, pytorch_pretrained_bert, nltk, spacy`<br>

10. [AllenNLP SRL](https://demo.allennlp.org/semantic-role-labeling) 설치<br>
    `pip install allennlp==1.0.0 allennlp-models==1.0.0`<br>

11. [Jsonnet](https://jsonnet.org/) 설치<br>
     `pip install jsonnet`<br>
     For Windows, `pip install jsonnetbin`<br>
   
12. ESL Annotator 패키지 설치<br>
    `git clone https://github.com/ihaeyong/drama-graph.git`<br>
    - script-esf 폴더<br>

## 사용법

- script-esf/generate_event_structure_lexicon.py에서 'ewiser_path'와 'ewiser_input' 폴더 경로를 [your_path]로 수정.<br>
- 실행 코드. <br>
  `python generate_event_structure_lexicon.py """your sentence"""`<br>

## 입출력 예시 설명

> 입력: 문장 텍스트 (예: John arrived in Seoul yesterday.)<br>
> 출력: 주석 결과 json파일 (script-esf/esl_annotation.result.json)<br>

1. **동사 텍스트 토큰과 레마**<br>
>v.text: arrived <br>
>v.lemma: arrive <br>

2. **동사 의미**(WSD by ewiser)<br>
> wn_synset: arrive.v.01<br>
> v.offset: wn:02005948v<br>
> v.sense_key: arrive%2:38:00:: <br>

3. **사건구조프레임 타입**(Event Structure Frame Type)
> MOVE_TO_GOAL <br>

4. **사건구조프레임**(Event Structure Frame; ESF)<br>
문장에서 동사가 나타내는 사건 전후의 변화를 포착하기 위해 <전상태(pre-state), 진행(process), 후상태(post-state)>로 구조화한  의미구조. <br>
아래 예시는 t1에 John이 도착점인 Seoul에 있지 않고, 출발점(source_location)에 있다가 arriving (t2)후 t3에 John은 Seoul에 있게 됨을 의미한다.<br>

> 사건구조프레임: <br>
> se1: pre-state: not_be (agent, at_goal_location, t1)<br>
> se2: pre-state: be (agent, at_source_location, t1)<br>
> se3: process: arriving (agent, at_goal_location, t2)<br>
> se4: post-state: be (agent, at_goal_location, t3)<br>
> se5: pos-state: not_be (agent, at_source_location, t3)<br>

5. **의미역**(Semantic Role) <br>
의미역은 문장에 표현된 사건의 참여자와 시간, 장소, 원인, 목적, 방법 등. <br>
다음은 AllenNLP SRL에 의해 주석된 의미역 예. ARG0은 참여자, ARGM-LOC은 장소, ARGM-TMP는 시간.<br>

> {"ARG0": "John", "VERB": "arrived", "ARGM-LOC": "in_Seoul", "ARGM-TMP": "yesterday"}<br>

6. **동의어와 상위어**
> synonyms: ['arrive', 'get', 'come']<br>
> hypernyms: []
