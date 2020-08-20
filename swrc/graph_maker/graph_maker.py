import requests
from graphviz import Digraph
import spacy
from utils.macro import *

def eng_frameBERT(text):
    host = 'http://143.248.135.188:1106/frameBERT'
    data = {
        'text': text
    }
    response = requests.post(host, data=data, verify=False)
    return response.json()

def virtuoso(text):
    host = 'http://kbox.kaist.ac.kr:1259/vtt_virtuoso'
    data = {
        'text': text.lower()
    }

    chars = ['dokyung', 'haeyoung1', 'haeyoung2', 'sukyung', 'jinsang', 'taejin', 'hun', 'jiya', 'kyungsu', 'deogi',
             'heeran', 'jeongsuk', 'anna', 'hoijang', 'soontack', 'sungjin', 'gitae', 'sangseok', 'yijoon', 'seohee']

    response = requests.post(host, data=data, verify=False)
    return response.json()

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
        self.build_graph_for_json()
        self.whole_graph = self.build_graph()
        print(' done..')
        if config['graph']['visualization'] != 'None':
            print(' visualizing..')
            self.visualization()
            print(' done..')


    def struct_knowledge(self):  # coref + 해당 지식이 인물관련 지식인가?
        def form_to_character(form, coref_dict):
            if form.lower() in coref_dict.keys():
                name = coref_dict[form.lower()]
                if name == 'Chairman' or name == 'Chairmna':
                    name = 'Hoijang'
                return True, name
            return False, form

        if self.config['mode'] == 'subtitle' or self.config['mode'] == 'demo':
            for ep in self.input:
                for scene in ep:
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


    def build_graph_for_json(self):
        txt = open(self.config['graph']['character_name'], 'r')
        lines = txt.readlines()
        txt.close()
        self.char_names = [name.strip() for name in lines]
        jsons = {}
        nlp = spacy.load('en_core_web_sm')

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

                exist_ch = set()
                for triple in triples:
                    if triple['subject'] in self.char_names:
                        exist_ch.add(triple['subject'])
                    if triple['object'] in self.char_names:
                        exist_ch.add(triple['object'])
                exist_ch = list(exist_ch)

                back_KB = []
                for ch in exist_ch: # extract back KB
                    knowledges = virtuoso(ch)
                    knowledges = knowledges[0]
                    for k in knowledges:
                        sbj = ch
                        rel = k['p'].split('/')[-1]
                        obj = k['o'].split('/')[-1]
                        t = {
                            'subject': sbj,
                            'relation': rel,
                            'object': obj
                        }
                        back_KB.append(t)

                args = []
                for t in triples:
                    if t['object'] not in self.char_names:
                        args.append(t['object'])
                refined_frames = []
                for f in frames:
                    t = {
                        'frame': f['frame'].split('#')[0],
                        'lu': f['lu'],
                        'args': []
                    }
                    for k,v in f.items():
                        if k == 'frame' or k == 'lu':
                            continue
                        t['args'].append({k:v})
                        args.append(v)
                    refined_frames.append(t)

                common_sense = []
                wiki = []
                done = []
                ch_list = ['dokyung', 'haeyoung1', 'haeyoung2', 'sukyung', 'jinsang', 'taejin', 'hun', 'jiya', 'kyungsu', 'deogi',
                 'heeran', 'jeongsuk', 'anna', 'hoijang', 'soontack', 'sungjin', 'gitae', 'sangseok', 'yijoon',
                 'seohee']


                for arg in args:
                    if arg.lower() in ch_list:
                        continue
                    sents = nlp(arg).sents
                    for sent in sents:
                        tokenized = [(tok.text, tok.pos_) for tok in sent]
                    if tokenized[-1][-1][0] == 'N':
                        sbj = tokenized[-1][0]

                        if sbj in done:
                            continue
                        if sbj.lower() in ch_list:
                            continue

                        cs_knowledges, wiki_knowledges = virtuoso(sbj)
                        for k in cs_knowledges:
                            rel = k['p'].split('/')[-1]

                            if rel.find("URL") > 0:
                                continue

                            obj = k['o'].split('/')[-1]
                            t = {
                                'subject': sbj,
                                'relation': rel,
                                'object': obj
                            }
                            common_sense.append(t)

                        for k in wiki_knowledges:
                            rel = k['p'].split('/')[-1]

                            if rel.find("URL") > 0:
                                continue

                            obj = k['o'].split('/')[-1]
                            t = {
                                'subject': sbj,
                                'relation': rel,
                                'object': obj
                            }
                            wiki.append(t)


                        done.append(sbj)


                json = {}
                json['char_background'] = back_KB
                json['common_sense'] = common_sense
                json['entity_background'] = wiki
                json['triples'] = triples
                json['frames'] = refined_frames
                jsons['ep{}_scene{}'.format(ep_id, s_id)] = json

        jsondump(jsons, self.config['graph']['json_path'])


    def build_graph(self):
        whole_graph = {}
        for name in self.char_names:
            whole_graph[name] = {'frame': [], 'triple': []}

        if self.config['mode'] == 'subtitle' or self.config['mode'] == 'demo':  # scene 당 그래프 1개.
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
                    self.graphs.append(('ep{}_scene{}'.format(ep_id, s_id), graph))



        # jsondump(self.graphs, self.config['graph']['json_path'])
        return whole_graph

    def visualization(self):
        # self.config['graph']['image_path']
        os.environ["PATH"] += os.pathsep + self.config['graph']['package_path']
        chars = self.config['graph']['character'].split(',')
        if chars[0] == 'all':
            chars = self.char_names


        for qid, graph in self.graphs:
            if self.config['graph']['visualization'] != 'frame':
                dot_triple = Digraph(comment='The Round Table', format='pdf')
            if self.config['graph']['visualization'] != 'triple':
                dot_frame = Digraph(comment='The Round Table', format='pdf')
            t_name_to_i = {}
            t_i_to_name = {}
            f_name_to_i = {}
            f_i_to_name = {}

            if self.config['graph']['visualization'] != 'frame':
                dot_triple = Digraph(comment='The Round Table')
                for char in chars:
                    triples = graph[char]['triple']


                    if len(triples) != 0:
                        if char not in t_name_to_i:
                            dot_triple.node(str(len(t_name_to_i)), char, _attributes={'fillcolor':'gray', 'style':'filled'})
                            t_name_to_i[char] = str(len(t_name_to_i))
                            t_i_to_name[str(len(t_name_to_i))] = char

                    for t in triples:
                        rel = str(t[0])
                        obj = str(t[1])
                        type = str(t[2])
                        if obj not in t_name_to_i:
                            if type == 'background':  # background
                                dot_triple.node(str(len(t_name_to_i)), obj)
                                # dot_triple.node(str(len(t_name_to_i)), obj, _attributes={'fillcolor':'blue', 'style':'filled'})
                            else:
                                dot_triple.node(str(len(t_name_to_i)), obj)
                            t_name_to_i[obj] = str(len(t_name_to_i))
                            t_i_to_name[str(len(t_name_to_i))] = obj
                        if type == 'background':  # background
                            dot_triple.edge(t_name_to_i[char], t_name_to_i[obj], label=rel,  _attributes={'fontcolor':'red','fillcolor':'red', 'style':'filled'})
                        else:
                            dot_triple.edge(t_name_to_i[char], t_name_to_i[obj], label=rel)

            if self.config['graph']['visualization'] != 'triple':
                dot_frame = Digraph(comment='The Round Table')
                for char in chars:
                    frames = graph[char]['frame']

                    if len(frames) != 0:
                        if char not in f_name_to_i:
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
                            if k =='frame': continue
                            dot_frame.edge(f_name_to_i[f['frame']], f_name_to_i[v], label=k)

            if self.config['graph']['visualization'] != 'frame':
                dot_triple.render(os.path.join(self.config['graph']['graph_path'], '{}_triple.gv'.format(qid)), view=False)
            if self.config['graph']['visualization'] != 'triple':
                dot_frame.render(os.path.join(self.config['graph']['graph_path'], '{}_frame.gv'.format(qid)), view=False)
