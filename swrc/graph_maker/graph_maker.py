# TODO frame 같은 애들끼리 그래프가 겹치는 버그가 있다.

import os
from graphviz import Digraph
from utils.macro import *

class graph_maker:
    def __init__(self, config, input, back_KB):
        self.config = config
        self.input = input
        self.back_KB = back_KB
        self.char_triples = []
        self.char_frames = []
        self.char_names = []
        self.struct_knowledge()
        self.graphs = []
        self.whole_graph = self.build_graph()
        if config['graph']['visualization']:
            self.visualization()


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
                for u in qa['utterances']:
                    for sent in u['sents']:
                        sent['char_triples'] = []
                        sent['char_frames'] = []

                        coref_dict = {}  # {form:character}

                        for coref in sent['corefs']:
                            coref_dict[coref['form'].lower()] = coref['coref']

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
                        frames = {}
                        for item in sent['frames']:
                            item = [x.split(':')[-1].split('-')[-1] for x in item]
                            if item[0] not in frames:
                                frames[item[0]] = {'type': item[0]}

                            if item[2][-1] == '.':
                                item[2] = item[2][:-1]
                            flag = flag or cur_flag
                            frames[item[0]][item[1]] = item[2]

                        for key in frames:
                            frame = frames[key]
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
            whole_graph[name] = {'undirected': [], 'directed': []}

        if self.config['mode'] == 'qa':  # qa 당 그래프 1개.
            for qa in self.input:
                graph = {}
                for name in self.char_names:
                    graph[name] = {'undirected': [], 'directed': []}

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

                        if self.config['graph']['only_use'] == 'None':
                            only_rel = []
                        else:
                            only_rel = self.config['graph']['only_use'].split(',')

                        if k[0] in only_rel:
                            triples.append(k_dict)

                for t in triples:
                    if t['subject'] not in self.char_names:
                        continue
                    whole_graph[t['subject']]['directed'].append((t['relation'], t['object']))
                    graph[t['subject']]['directed'].append((t['relation'], t['object']))

                for f in frames:
                    for k, v in f.items():
                        if v in self.char_names:
                            whole_graph[v]['undirected'].append(f)
                            graph[v]['undirected'].append(f)
                self.graphs.append((qa['qid'], graph))
        jsondump(self.graphs, 'test.json')
        return whole_graph

    def visualization(self):
        # self.config['graph']['image_path']
        os.environ["PATH"] += os.pathsep + self.config['graph']['package_path']
        chars = self.config['graph']['character'].split(',')
        if chars[0] == 'all':
            chars = self.char_names


        for qid, graph in self.graphs:
            dot_triple = Digraph(comment='The Round Table')
            dot_frame = Digraph(comment='The Round Table')
            t_name_to_i = {}
            t_i_to_name = {}
            f_name_to_i = {}
            f_i_to_name = {}
            
            
            for char in chars:
                triples = graph[char]['directed']
                frames = graph[char]['undirected']

                
                if len(triples) != 0:
                    if char not in t_name_to_i:
                        dot_triple.node(str(len(t_name_to_i)), char, _attributes={'fillcolor':'gray', 'style':'filled'})
                        t_name_to_i[char] = str(len(t_name_to_i))
                        t_i_to_name[str(len(t_name_to_i))] = char

                for t in triples:
                    rel = str(t[0])
                    obj = str(t[1])
                    if obj not in t_name_to_i:
                        dot_triple.node(str(len(t_name_to_i)), obj)
                        t_name_to_i[obj] = str(len(t_name_to_i))
                        t_i_to_name[str(len(t_name_to_i))] = obj

                    dot_triple.edge(t_name_to_i[char], t_name_to_i[obj], label=rel)

            for char in chars:
                frames = graph[char]['undirected']

                if len(frames) != 0:
                    if char not in t_name_to_i:
                        dot_frame.node(str(len(f_name_to_i)), char, _attributes={'fillcolor':'gray', 'style':'filled'})
                        f_name_to_i[char] = str(len(f_name_to_i))
                        f_i_to_name[str(len(f_name_to_i))] = char

                for f in frames:
                    for k,v in f.items():
                        if v not in f_name_to_i:
                            dot_frame.node(str(len(f_name_to_i)), v)
                            f_name_to_i[v] = str(len(f_name_to_i))
                            f_i_to_name[str(len(f_name_to_i))] = v

                    for k, v in f.items():
                        if k =='type': continue
                        dot_frame.edge(f_name_to_i[f['type']], f_name_to_i[v], label=k)



            print(dot_triple.source)
            dot_triple.render(os.path.join(self.config['graph']['graph_path'], '{}_triple.gv').format(qid), view=False)
            dot_frame.render(os.path.join(self.config['graph']['graph_path'], '{}_frame.gv').format(qid), view=False)
            print()






