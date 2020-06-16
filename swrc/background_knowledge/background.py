import os
from utils.macro import *

def split_seq(seq, sep):
    start = 0
    while start < len(seq):
        try:
           stop = start + seq[start:].index(sep)
           yield seq[start:stop]
           start = stop + 1
        except ValueError:
           yield seq[start:]
           break

class background_knowledge:
    def __init__(self, config):
        self.config = config

        if config['background']['load']:
            self.output = self.load_KB()
        else:
            self.output = self.make_KB()

    def make_KB(self):
        txt = open(self.config['background']['input_path'], 'r')
        lines = txt.readlines()
        lines = [l.strip() for l in lines]
        txt.close()
        chars = [i for i in split_seq(lines, "")]

        kb = {}
        for char in chars:
            name = char[0]
            rels = []
            for i, rel in enumerate(char):
                if i == 0: continue
                rels.append(rel.split('#'))
            kb[name] = rels

        jsondump(kb, self.config['background']['output_path'])
        return kb

    def load_KB(self):
        return jsonload(self.config['background']['output_path'])
