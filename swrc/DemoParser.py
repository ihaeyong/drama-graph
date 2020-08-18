import yaml
import os
import logging
from preprocessing.preprocessor import *
from knowledge_extraction.extractor import *
from graph_maker.demo_graph_maker import *
from background_knowledge.background import *
from graphviz import Digraph
from stanfordcorenlp import StanfordCoreNLP
import requests

def virtuoso(text):
    host = 'http://kbox.kaist.ac.kr:1259/vtt_virtuoso'
    data = { "text": text.lower() }

    chars = ['dokyung', 'haeyoung1', 'haeyoung2', 'sukyung', 'jinsang', 'taejin', 'hun', 'jiya', 'kyungsu', 'deogi',
             'heeran', 'jeongsuk', 'anna', 'hoijang', 'soontack', 'sungjin', 'gitae', 'sangseok', 'yijoon', 'seohee']

    if text.lower() not in chars:
        return {}

    response = requests.post(host, data=data, verify=False)
    return response.json()


class Parser():
    def __init__(self):
        self.nlp_parser = None
        with open(os.path.join('config', 'demo.yaml'), 'r', encoding='utf8') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)


    def text2input(self, utters):
        origin = utters
        output = [{'scene':[], 'scene_number':"-1"}]

        for u in utters:
            speaker, text = u.split(':')
            speaker = speaker.strip()
            text = text.strip()

            utter = {
                'utter': text,
                'speaker': speaker,
                'scene_num': '-1'
            }

            if speaker == 'example':
                id = int(text)
                j = jsonload('data/demo_example/AnotherMissOh_ep01.json')
                output[0] = j[id]
                origin = []
                for u in output[0]['scene']:
                    tt = u['speaker'] + ':' + u['utter']
                    origin.append(tt)


            output[0]['scene'].append(utter)


        return origin, output


    def visualization(self, mode, graph):
        # self.config['graph']['image_path']
        chars = list(graph.keys())
        dot_triple = Digraph(comment='The Round Table')
        dot_frame = Digraph(comment='The Round Table')
        t_name_to_i = {}
        t_i_to_name = {}
        f_name_to_i = {}
        f_i_to_name = {}

        if mode != 'frame':
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

        if mode != 'triple':
            for char in chars:
                frames = graph[char]['frame']
                backs = virtuoso(char)

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


                for back in backs:
                    p = back['p'].split('/')[-1]
                    o = back['o'].split('/')[-1]
                    if o not in f_name_to_i.keys():
                        dot_frame.node(str(len(f_name_to_i)), o)
                        f_name_to_i[o] = str(len(f_name_to_i))
                        f_i_to_name[str(len(f_name_to_i))] = o

                    dot_frame.edge(f_name_to_i[char], f_name_to_i[o], label=p, _attributes={'fontcolor':'red','fillcolor':'red', 'style':'filled'})

        return dot_triple.source, dot_frame.source


    def parser(self, input, mode):
        origin, script = self.text2input(input)
        config = self.config
        if mode == 'triple':
            config['extraction']['oie'] = 'stanford'
            config['extraction']['frame'] = 'None'
        elif mode == 'frame':
            config['extraction']['oie'] = 'None'
            config['extraction']['frame'] = 'frameBERT'
        else:
            config['extraction']['oie'] = 'stanford'
            config['extraction']['frame'] = 'frameBERT'

        _preprocessor = preprocessor(config, script)
        _extractor = extractor(config, _preprocessor.output, self.nlp_parser)
        _back_KB = background_knowledge(config)
        _graph_maker = graph_maker(config, _extractor.output, _back_KB.output)

        json_graph = _graph_maker.whole_graph
        del_keys = []
        for k,v in json_graph.items():
            if len(v['triple']) == 0 and len(v['frame']) == 0:
                del_keys.append(k)
        for k in del_keys:
            del json_graph[k]

        output = {'origin': origin}
        us = _preprocessor.output[0][0]['scene']
        output['utters'] = []
        for u in us:
            for s in u['sents']:
                del s['statement']
                del s['frames']
                del s['char_frames']
                del s['triples']
                del s['char_triples']
            output['utters'].append(u['sents'])


        output['json_graph'] = json_graph

        output['triple_graph'], output['frame_graph'] = self.visualization(mode, json_graph)

        print()



        return output
