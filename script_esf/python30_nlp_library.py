# -*- coding: utf-8 -*-

import re
import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


pronoun = ['i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves', 'they', 'them', 'their', 'theirs', 'themselves']
auxiliary = ['can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', "'ll", "'d", "ca", "wo"]
be_verb = ['am', "'m", "was", "are", "'re", "were", "is", "'s", "be", "being", "been"]
contracted = ["'m", "'s", "'ll", "'ve", "'d", "n't"]
gerund = ['wedding']


def file_open(File):
    f = open(File, 'r', encoding='UTF-8')
    data = f.readlines()
    return data


def string_to_file(string, File):
    with open(File, 'w', encoding='UTF-8') as f:
        f.write(string)
    f.close()
    

def dict_to_json(my_dict, my_json_file):
    with open(my_json_file, 'w', encoding='UTF-8') as json_file:
        json.dump(my_dict, json_file)
    

def json_to_dict(my_json_file):
    with open(my_json_file, 'r', encoding='UTF-8') as json_file:
        data = json.load(json_file)
    return data
    

def sentence_tokenize(line):
    sents=[]
    if line.strip() != "":
        line = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', line.strip())
        line = re.sub(r'([a-z])\?([A-Z])', r'\1? \2', line.strip())
        line = re.sub(r'([a-z])\!([A-Z])', r'\1! \2', line.strip())

        sents = sent_tokenize(line.strip())
    return sents


def tokenize(sentence):
    tokens = word_tokenize(sentence)
    return tokens


def pos_tagging(sentence):
    tokens = word_tokenize(sentence)
    pos_tagged = nltk.pos_tag(tokens)
    return pos_tagged


def lemmatize(word):
    lemma=word
    infl_dict = json_to_dict('vInflection.json')
    for infl in infl_dict:
        if word in infl['text_token']:
            lemma = infl['lemma']
        else: pass
    return lemma


def has_verb(sentence):
    answer = False
    pos_tagged_sentence = pos_tagging(sentence)
    #print(pos_tagged_sentence)
    for w in pos_tagged_sentence:
        if w[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            answer = True
        else:
            pass
    return answer


def is_pronoun(word):
    answer = False
    if word in pronoun:
        answer = True
    else: pass
    return answer


def is_be_verb(word):
    answer = False
    if word in be_verb:
        answer = True
    else: pass
    return answer


def is_auxiliary(word):
    answer = False
    if word in auxiliary:
        answer = True
    else: pass
    return answer


def is_contracted_form(word):
    answer = False
    if word in contracted:
        answer = True
    else: pass
    return answer


def is_gerund(word):
    answer = False
    if word in gerund:
        answer = True
    else: pass
    return answer


def remove_text_in_parentheses(string):
    paren = re.compile('\(.+?\)')
    output = paren.sub('', string)
    return output



        
        
        
    
