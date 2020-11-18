#-*- coding: utf-8 -*-

import re
import sys
import io
from subprocess import Popen, PIPE, STDOUT
import json
from python30_nlp_library import lemmatize



class ewiserWrapper:

    def __init__(self, path, input_file):
        print("\n\ninitializing ewiserWrapper--init()\n")
        
        cmd = ["python", "bin/annotate.py", "-c", "ewiser.semcor+wngt.pt", input_file]
        self.proc = Popen(cmd, cwd=path, stdin=PIPE, stdout = PIPE, shell=True, universal_newlines=True)

    def process(self, input_file):
        self.give_input(input_file)
        return self.get_output()

    def give_input(self, input_file):
        self.proc.stdin.flush()
        self.proc.stdin.write(input_file)

    def get_output(self):
        data = self.proc.stdout.readlines()
        return data
    
    def ewiser_to_origin(self, ewiser_annotated):
        tokens = []
        sent = ewiser_annotated.split(' ')
        for s in sent:
            tokens.append(s.split('@#*')[0])
        text = ' '.join(tokens).split(' .')[0]+'.'
        return text

    def ewiser_tokenize(self, ewiser_annotated):
        middle = ewiser_annotated.split(' ')
        token_list = []
        num = 1
        for m in middle:
            if m == '\n' or m == '': pass
            else:
                m_tokens = m.split('@#*')
                token = {}
                token['wid']= num
                num += 1
                token['token']=m_tokens[0]
                token['lemma']=lemmatize(m_tokens[1])
                token['pos']=m_tokens[2]
                token['offset']=m_tokens[3]
                token_list.append(token)
        return token_list
            
    def ewiser_result_to_dictionary(self, string):
        line_dict = {}
        if string.startswith('Namespace') or string=='\n': pass
        else:
            line_dict['text']=self.ewiser_to_origin(string)
            line_dict['words']=self.ewiser_tokenize(string)
        return line_dict

    def get_verb_sense(self, input_file):
        ewiser_result = self.process(input_file)
        verb_lexicon=[]
        for output in ewiser_result:   
            if output == '' and self.proc.poll() is not None:
                break
            if output:
                result=output.strip()
                result_dict=self.ewiser_result_to_dictionary(result)
                
                if result_dict != {}:
                    v_lexicon=[]
                    for w in result_dict['words']:
                        if w['pos']=='VERB':
                            v_dict={}
                            v_dict['v.id']= w['wid']
                            v_dict['v.text']= w['token']
                            v_dict['v.lemma']= w['lemma']
                            v_dict['v.offset']= w['offset']
                            v_lexicon.append(v_dict)
                    #print(v_lexicon)
                    verb_lexicon.extend(v_lexicon)
        return verb_lexicon

    def quit(self):
        self.proc.terminate()

               




            

