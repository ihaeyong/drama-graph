from knowledge_extraction.oie import *
from knowledge_extraction.frame import *
from preprocessing.sentence_divider import *
from utils.macro import *

class extractor:
    def __init__(self, config, input):
        self.input = input
        self.config = config
        self.oie = oie(self.input, self.config['preprocessing']['oie'])
        self.frame = frame(self.oie.output, self.config['preprocessing']['frame'])
        self.output = self.frame.output


    def subtitle_loader(self):
        return

    def qa_loader(self):
        qa_path = self.config['preprocessing']['qa_file']
        self.input = jsonload(qa_path)
        print()

    def save_output(self):
        jsondump(self.output, os.path.join(self.config['preprocessing']['output_path'], 'extracted.json'))
        return

