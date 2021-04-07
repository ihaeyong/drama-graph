# -*- coding: utf-8 -*-


import sys
import io
from ewiserWrapper import ewiserWrapper
from allennlp.predictors.predictor import Predictor
from python30_nlp_library import json_to_dict, dict_to_json, sentence_tokenize, string_to_file, is_auxiliary, is_gerund
from esf_lib import esfs


ewiser_path="C:/Users/PC2/ewiser/"
ewiser_input = "C:/Users/PC2/script_esf/ewiser_input.txt"

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")
esfwn = json_to_dict('esfwn_v1.json')




# Word Sense Disambiguation by EWISER
def wsd(sentence):
    string_to_file(sentence, ewiser_input)
    wsd = ewiserWrapper(ewiser_path, ewiser_input)
    result = wsd.get_verb_sense(ewiser_input)
    return result



def list_args(srl_sent):
    args = []
    chunks = srl_sent.split('] ')
    for ch in chunks:
        if '[' in ch:
            new_ch=ch.split('[')[1]
            if ': ' in new_ch:
                Tuple = new_ch.split(': ')
                srtag_arg = (Tuple[0], Tuple[1].replace(']', ''))
                args.append(srtag_arg)
            else: pass
        else: pass    
    return args


# Semantic Role Labeling by allenNLP SRL
def get_semantic_roles(sent):
    sem_roles = []
    srls=predictor.predict(sentence=sent)
    for verb in srls['verbs']:
        if is_auxiliary(verb['verb'].lower()) or is_gerund(verb['verb'].lower()): pass
        else:
            verb_args = list_args(verb['description'])
            sem_roles.append(verb_args)
    return sem_roles


def get_vtext(sem_role):
    vtext = ''
    for member in sem_role:
        if member[0]=='V':
            vtext = member[1]
        else: pass
    return vtext
            

# merge wsd result with srl result
def merge_wsd_with_srl(wsd_dict_list, srl_list):
    merged_list = []
    for v1 in wsd_dict_list:
        for v2 in srl_list:
            vtext = get_vtext(v2)
            if v1['v.text']==vtext:
                v1['v.sem_roles']=v2
                merged_list.append(v1)
    return merged_list


# link wsd result to esfwn and get synset number, esf_type, synonyms, hypernyms
def link_wsd_with_esfwn(wsd_dict_list, esf_type_list):
    linked_list = []
    for v1 in wsd_dict_list:
        for v2 in esf_type_list:
            if v1['v.lemma']==v2['VERB'] and v1['v.offset']==v2['OFFSET']:
                v1['wn_synset']=v2['SENSE_NUMBER']
                v1['esf_type']=v2['ESF_TYPE']
                v1['synonyms']=v2['SYNONYMS']
                v1['hypernyms']=v2['HYPERNYMS']
                linked_list.append(v1)
    return linked_list


def remove_duplicates_from_dict_list(dictlist):
    seen = set()
    new_list = []
    for dic in dictlist:
        if str(dic['v.id']) not in seen:
            seen.add(str(dic['v.id']))
            new_list.append(dic)
        else: pass
    return new_list
    

def get_verb_esl(sentence):
    eslexicon = {}
    eslexicon['sentence']=sentence

    print("Word Sense Disambiguation...")
    verb_sense_dict = wsd(sentence) # keys = ['v.id', 'v.text', 'v.lemma', 'v.offset']

    print("Semantic Role Labelling...")
    verb_sem_roles = get_semantic_roles(sentence) # added_key = 'v.sem_roles'
    
    wsd_with_srl = merge_wsd_with_srl(verb_sense_dict, verb_sem_roles)

    print("ESF-type annotating...")
    wsd_with_esf = link_wsd_with_esfwn(wsd_with_srl, esfwn)
    eslexicon['verbs']=wsd_with_esf
    for v in eslexicon['verbs']:
        for esf in esfs:
            if v['esf_type']==esf['etype']:
                v['esf']=esf['esf']

    new_esl = remove_duplicates_from_dict_list(eslexicon['verbs'])
    eslexicon['verbs']=new_esl
    return eslexicon


sentence = sys.argv[1]

esl = get_verb_esl(sentence)
dict_to_json(esl, 'esl_annotation_result.json')
