from knowledge_extraction.oie import *
from knowledge_extraction.frame import *
from preprocessing.sentence_processor import *
from utils.macro import *

class extractor:
    def __init__(self, config, input, nlp_parser=None):
        self.input = input
        self.config = config
        self.nlp_parser = nlp_parser

        if config['extraction']['load']:
            self.output = jsonload(self.config['extraction']['output_path'])
            return
        self.frame = frame(config, self.input)
        self.oie = oie(config, self.frame.output, nlp_parser)
        self.output = self.oie.output
        # self.output = self.frame.output

    def save_output(self):
        if self.config['extraction']['load']:
            return
        jsondump(self.output, self.config['extraction']['output_path'])
        return

