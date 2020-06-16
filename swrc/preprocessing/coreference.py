import re
import os

class coreference:
    def __init__(self, config, input):
        self.config = config
        self.input = input

        if config['preprocessing']['coreference'] == 'gold': # use annotations
            self.output = self.coref_gold()

        else:  # use models
            print()

    def coref_gold(self):
        input = self.input
        coref_p = re.compile(r'[(]\w+[)]')
        if self.config['mode'] == 'qa':
            for qa in input:
                for u in qa['utterances']:
                    words = u['old_utter'].split()
                    new_words = []
                    corefs = []
                    prefix = 0
                    for i, w in enumerate(words):
                        patts = coref_p.findall(w)
                        new_w = re.sub(coref_p, '', w)
                        new_words.append(new_w)
                        begin = prefix
                        for patt in patts:
                            idx = w.find(patt)

                            coref = {
                                'word_id': i,
                                'begin': begin,
                                'end': begin + idx,
                                'form': '',
                                'coref': patt[1:-1]
                            }

                            corefs.append(coref)
                        prefix += len(new_w) + 1

                    u['utter'] = ' '.join(new_words)
                    for coref in corefs:
                        coref['form'] = u['utter'][coref['begin']:coref['end']]
                    u['corefs'] = corefs
        elif self.config['mode'] == 'subtitle':
            for j in input:
                for scene in j:
                    for u in scene['scene']:
                        u['old_utter'] = u['utter']
                        words = u['old_utter'].split()
                        new_words = []
                        corefs = []
                        prefix = 0
                        for i, w in enumerate(words):
                            patts = coref_p.findall(w)
                            new_w = re.sub(coref_p, '', w)
                            new_words.append(new_w)
                            begin = prefix
                            for patt in patts:
                                idx = w.find(patt)

                                coref = {
                                    'word_id': i,
                                    'begin': begin,
                                    'end': begin + idx,
                                    'form': '',
                                    'coref': patt[1:-1]
                                }

                                corefs.append(coref)
                            prefix += len(new_w) + 1

                        u['utter'] = ' '.join(new_words)
                        for coref in corefs:
                            coref['form'] = u['utter'][coref['begin']:coref['end']]
                        u['corefs'] = corefs
        return input