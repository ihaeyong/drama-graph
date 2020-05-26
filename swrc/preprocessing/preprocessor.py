from knowledge_extraction.oie import *
from knowledge_extraction.frame import *
from preprocessing.sentence_divider import *
from utils.macro import *

class preprocessor:
    def __init__(self, config, mode):
        self.input = []
        self.config = config
        if mode == 'subtitle':
            self.subtitle_loader()
        else:
            self.qa_loader()
        self.sentence_divider = sentence_divider(self.input)
        self.output = self.sentence_divider.output


    def subtitle_loader(self):
        return

    def qa_loader(self):
        qa_path = self.config['preprocessing']['qa_file']
        self.input = jsonload(qa_path)

    def save_output(self):
        jsondump(self.output, os.path.join(self.config['preprocessing']['output_path'], 'preprocessed.json'))
        return

