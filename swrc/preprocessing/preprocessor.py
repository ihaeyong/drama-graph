from preprocessing.sentence_divider import *
from preprocessing.coreference import *
from preprocessing.to_statement import *
from utils.macro import *

class preprocessor:
    def __init__(self, config):
        self.input = []
        self.config = config
        if config['preprocessing']['load']:
            self.output = jsonload(self.config['preprocessing']['output_path'])
            return
        elif config['mode'] == 'subtitle':
            self.subtitle_loader()
        else:
            self.qa_loader()

        self.coref = coreference(config, self.input)
        self.sentence_divider = sentence_divider(config, self.coref.output)
        self.to_stmt = to_statement(config, self.sentence_divider.output)
        self.output = self.to_stmt.output


    def subtitle_loader(self):
        subtitle_path = self.config['preprocessing']['substitle_file']
        for path in diriter(subtitle_path):
            self.input.append(jsonload(path))
        return

    def qa_loader(self):
        qa_path = self.config['preprocessing']['qa_file']
        self.input = jsonload(qa_path)

    def save_output(self):
        if self.config['preprocessing']['load']:
            return
        jsondump(self.output, self.config['preprocessing']['output_path'])
        return
