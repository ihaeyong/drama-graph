# ESFWN-based ESL Annotator
**Event Structure Frame-annotated WordNet (ESFWN)** is a knowledge base which includes the offset and the synset number of WordNet and the proper Event Structure Frames (ESF) for all verbs in English. The ESFWN makes it possible to use the information WordNet provides and the ESF for verbs in English text. Its filename is **esfwn_v1.json**.       
**ESFWN-based ESL Annotator** annotates the Event Structure Lexicon (ESL) for each verb in English text. The ESL includes verb itself, verb lemma, the WordNet offset and synset number for linking to WordNet, synonyms and hypernyms, the ESF type and its corresponding ESF.


## ESL Example
   **sentence**: John is **putting** a book on a table.    
   **v.text**: putting    
   **v.lemma**: put    
   **v.offset**: wn:01494310v    
   **v.sem_roles**: [(ARG0, John), (V, put), (ARG1, a book), (ARG2, on a table)]    
   **wn_synset**: put.v.01    
   **synonyms**: [put, set, place, pose, position, lay]    
   **hypernyms**: [move, displace]    
   **esf_type**: CAUSE_MOVE_TO_GOAL    
   **esf**: [{se_num: se1, time: t1, se_type: pre-state, se_form: be (theme, source_location)},    
              {se_num: se2, time: t1, se_type: d-pre-state, se_form: be_not (theme, goal_location)},    
              {se_num: se3, time: t2, se_type: process, se_form: V-ing (agent, source_location, goal_location)},    
              {se_num: se4, time: t3, se_type: post-state, se_form: be (theme, goal_location)}]    
              (**pre-state**: a presupposed state before the maintaining event, **post-state**: an entailed state after the maintaining event)

## Pre-requisite and Installation

#### 1. EWISER(https://github.com/SapienzaNLP/ewiser)
      - Follow the installation guideline the EWISER provides. 
      - **External Downloads**: 
        You should download [SemCor + tagged glosses + WordNet Examples](https://drive.google.com/file/d/11RyHBu4PwS3U2wOk-Le9Ziu8R3Hc0NXV/view?usp=sharing) and unzip and put it into the ewiser folder.

#### 2. AllenNLP SRL(https://demo.allennlp.org/semantic-role-labeling)
      pip install allennlp==1.0.0 allennlp-models==1.0.0

#### 3. install NLTK (Natural Language ToolKit) and NLTK Dataset
      - pip install nltk
      - install NLTK Dataset
       ```python
        >>>import nltk
        >>>nltk.download()
       ```

#### 4. git clone https://github.com/ish97/ESFWN-based-ESL-Annotator.git


## How To Use
#### 1. Change 'ewiser_path' and 'ewiser_input' path to [your path] in the 'generate_event_structure_lexicon.py'
#### 2. **Usage**: python generate_event_structure_lexicon.py "your sentence"
#### 3. **Check the output**: 'esl_annotation.result.json'

