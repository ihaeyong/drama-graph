import yaml
import os
import logging
from preprocessing.preprocessor import *
from knowledge_extraction.extractor import *
from graph_maker.demo_graph_maker import *
from background_knowledge.background import *


class Parser():
    def __init__(self):
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
        _extractor = extractor(config, _preprocessor.output)
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


        output['graph'] = json_graph
        return output
