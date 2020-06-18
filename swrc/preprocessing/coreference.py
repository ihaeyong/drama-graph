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
        coref_p = re.compile(r'[(].+[)]')
        if self.config['mode'] == 'qa':
            for qa in input:
                for u in qa['utterances']:
                    u['utter'] = u['utter'].replace('}', ')')
                    u['utter'] = u['utter'].replace('((', '(')
                    u['utter'] = u['utter'].replace('))', ')')
                    u['old_utter'] = u['utter']

                    open = []
                    close = []
                    patts = []
                    corefs = []

                    for i, char in enumerate(u['old_utter']):
                        if char == '(':
                            open.append(i)
                        if char == ')':
                            close.append(i)

                    if len(open) != len(close):
                        print('error')
                        print(u['utter'])
                        u['corefs'] = []
                        continue

                    for i in range(len(open)):
                        patt = u['old_utter'][open[i]:close[i] + 1]
                        patts.append(patt)
                        try:
                            mention = u['old_utter'][:open[i]].split()[-1]
                        except:
                            patts = []
                            corefs = []
                            break
                        st = open[i] - len(mention)
                        en = open[i]
                        coref = {
                            'begin': st,
                            'end': en,
                            'form': u['old_utter'][st:en],
                            'coref': patt[1:-1]
                        }

                        corefs.append(coref)

                    new_sent = u['old_utter']
                    idx = 0
                    for ii, patt in enumerate(patts):
                        new_sent = new_sent.replace(patt, '')
                        corefs[ii]['begin'] -= idx
                        corefs[ii]['end'] -= idx
                        idx += len(patt)
                    u['utter'] = new_sent
                    u['corefs'] = corefs
        elif self.config['mode'] == 'subtitle':
            for j in input:
                for scene in j:
                    for u in scene['scene']:
                        u['utter'] = u['utter'].replace('}', ')')
                        u['utter'] = u['utter'].replace('((','(')
                        u['utter'] = u['utter'].replace('))', ')')
                        u['old_utter'] = u['utter']

                        open = []
                        close = []
                        patts = []
                        corefs = []

                        for i, char in enumerate(u['old_utter']):
                            if char == '(':
                                open.append(i)
                            if char == ')':
                                close.append(i)

                        if len(open) != len(close):
                            print('error')
                            print(u['utter'])
                            u['corefs'] = []
                            continue

                        for i in range(len(open)):
                            patt = u['old_utter'][open[i]:close[i] + 1]
                            patts.append(patt)
                            try:
                                mention = u['old_utter'][:open[i]].split()[-1]
                            except:
                                patts = []
                                corefs = []
                                break
                            st = open[i] - len(mention)
                            en = open[i]
                            coref = {
                                'begin': st,
                                'end': en,
                                'form': u['old_utter'][st:en],
                                'coref': patt[1:-1]
                            }

                            corefs.append(coref)

                        new_sent = u['old_utter']
                        idx = 0
                        for ii, patt in enumerate(patts):
                            new_sent = new_sent.replace(patt, '')
                            corefs[ii]['begin'] -= idx
                            corefs[ii]['end'] -= idx
                            idx += len(patt)
                        u['utter'] = new_sent
                        u['corefs'] = corefs
        return input