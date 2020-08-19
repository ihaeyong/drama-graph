import os
from utils.macro import *

class graph_maker:
    def __init__(self, config, input, back_KB):
        self.config = config
        self.input = input
        self.back_KB = back_KB
        self.char_triples = []
        self.char_frames = []
        self.char_names = []
        print(' struct knowledge..')
        self.struct_knowledge()
        print(' done..')
        self.graphs = []
        print(' make graph..')
        self.whole_graph = self.build_graph()
        print(' done..')


    def struct_knowledge(self):  # coref + 해당 지식이 인물관련 지식인가?
        def form_to_character(form, coref_dict):
            if form.lower() in coref_dict.keys():
                name = coref_dict[form.lower()]
                if name == 'Chairman' or name == 'Chairmna':
                    name = 'Hoijang'
                return True, name
            return False, form

        if self.config['mode'] == 'qa':
            for qa in self.input:
                for uid, u in enumerate(qa['utterances']):

                    coref_dict = {}  # {form:character}
                    for coref in u['corefs']:
                        coref_dict[coref['form'].lower()] = coref['coref']

                    for sid, sent in enumerate(u['sents']):
                        sent['char_triples'] = []
                        sent['char_frames'] = []
                        if 'triples' not in sent.keys():
                            sent['triples'] = []
                        if 'frames' not in sent.keys():
                            sent['frames'] = []

                        for triple in sent['triples']:  # for directed edge
                            keys = triple.keys()
                            flag = False
                            for key in keys:
                                cur_flag, triple[key] = form_to_character(triple[key], coref_dict)
                                flag = flag or cur_flag
                            if flag:
                                sent['char_triples'].append(triple)

                        if type(sent['frames']) is dict:  # error case
                            continue
                        for id, frame in enumerate(sent['frames']):
                            frame['frame'] += '#{}_{}_{}'.format(uid,sid,id)
                            flag = False
                            for k in frame:
                                cur_flag, frame[k] = form_to_character(frame[k], coref_dict)
                                flag = flag or cur_flag
                            if flag:
                                sent['char_frames'].append(frame)

        elif self.config['mode'] == 'subtitle' or self.config['mode'] == 'demo':
            for ep in self.input:
                for scene in ep:
                    if scene['scene_number'] == 24:
                        print()
                    for uid, u in enumerate(scene['scene']):
                        coref_dict = {}  # {form:character}
                        for coref in u['corefs']:
                            coref_dict[coref['form'].lower()] = coref['coref']

                        for sid, sent in enumerate(u['sents']):
                            sent['char_triples'] = []
                            sent['char_frames'] = []

                            if 'triples' not in sent.keys():
                                sent['triples'] = []
                            if 'frames' not in sent.keys():
                                sent['frames'] = []


                            for triple in sent['triples']:  # for directed edge
                                keys = triple.keys()
                                flag = False
                                for key in keys:
                                    cur_flag, triple[key] = form_to_character(triple[key], coref_dict)
                                    flag = flag or cur_flag
                                if flag:
                                    sent['char_triples'].append(triple)


                            if type(sent['frames']) is dict:  # error case
                                continue
                            for id, frame in enumerate(sent['frames']):
                                frame['frame'] += '#{}_{}_{}'.format(uid,sid,id)
                                flag = False
                                for k in frame:
                                    cur_flag, frame[k] = form_to_character(frame[k], coref_dict)
                                    flag = flag or cur_flag
                                if flag:
                                    sent['char_frames'].append(frame)


    def build_graph(self):
        whole_graph = {}

        txt = open(self.config['graph']['character_name'], 'r')
        lines = txt.readlines()
        txt.close()
        self.char_names = [name.strip() for name in lines]
        for name in self.char_names:
            whole_graph[name] = {'frame': [], 'triple': []}

        if self.config['mode'] == 'qa':  # qa 당 그래프 1개.
            for qa in self.input:
                graph = {}
                for name in self.char_names:
                    graph[name] = {'frame': [], 'triple': []}

                triples = [triple for u in qa['utterances'] for sent in u['sents'] for triple in sent['char_triples']]
                frames = [triple for u in qa['utterances'] for sent in u['sents'] for triple in sent['char_frames']]

                for char in self.back_KB:
                    ks = self.back_KB[char]
                    for k in ks:
                        k_dict = {
                            'subject': char,
                            'relation': k[0],
                            'object': k[1]
                        }

                        if self.config['graph']['only_use'] == 'None':  #특정인물만.
                            only_rel = []
                        else:
                            only_rel = self.config['graph']['only_use'].split(',')

                        if k[0] in only_rel:
                            triples.append(k_dict)

                for t in triples:
                    if t['subject'] not in self.char_names:
                        continue
                    whole_graph[t['subject']]['triple'].append((t['relation'], t['object']))
                    graph[t['subject']]['triple'].append((t['relation'], t['object']))

                for f in frames:
                    for k, v in f.items():
                        if v in self.char_names:
                            whole_graph[v]['frame'].append(f)
                            graph[v]['frame'].append(f)
                self.graphs.append((qa['qid'], graph))

        elif self.config['mode'] == 'subtitle' or self.config['mode'] == 'demo':  # scene 당 그래프 1개.
            for i, ep in enumerate(self.input):
                ep_id = i+1
                for scene in ep:
                    graph = {}
                    s_id = scene['scene_number']
                    us = scene['scene']

                    for name in self.char_names:
                        graph[name] = {'frame': [], 'triple': []}

                    triples = [triple for u in us for sent in u['sents'] for triple in
                               sent['char_triples']]
                    frames = [triple for u in us for sent in u['sents'] for triple in sent['char_frames']]
                    backs = []

                    exist_ch = set()
                    for triple in triples:
                        if triple['subject'] in self.char_names:
                            exist_ch.add(triple['subject'])
                        if triple['object'] in self.char_names:
                            exist_ch.add(triple['object'])
                    exist_ch = list(exist_ch)




                    for char in self.back_KB:
                        if self.config['graph']['use_backKB'] == False:
                            break
                        if char not in exist_ch:
                            continue
                        ks = self.back_KB[char]
                        for k in ks:
                            k_dict = {
                                'type': 'background',
                                'subject': char,
                                'relation': k[0],
                                'object': k[1]
                            }
                            triples.append(k_dict)
                            backs.append(k_dict)

                    for t in triples:
                        if t['subject'] not in self.char_names:
                            continue

                        if 'type' in t.keys():
                            type = 'background'
                        else:
                            type = 'triple'

                        whole_graph[t['subject']]['triple'].append((t['relation'], t['object'], type))
                        graph[t['subject']]['triple'].append((t['relation'], t['object'], type))

                    for f in frames:
                        for k, v in f.items():
                            if v in self.char_names:
                                whole_graph[v]['frame'].append(f)
                                graph[v]['frame'].append(f)

                    # for back in backs:


                    self.graphs.append(('ep{}_scene{}'.format(ep_id, s_id), graph))



        jsondump(self.graphs, self.config['graph']['json_path'])
        return whole_graph
