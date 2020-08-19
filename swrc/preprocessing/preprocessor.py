from preprocessing.sentence_processor import *
from preprocessing.coreference import *
from preprocessing.to_statement import *
from utils.macro import *

class preprocessor:
    def __init__(self, config, demo_json):
        self.input = []
        self.config = config
        if config['preprocessing']['load']:
            self.output = jsonload(self.config['preprocessing']['output_path'])
            return
        elif config['mode'] == 'subtitle':
            self.subtitle_loader()
        elif config['mode'] == 'demo':
            self.demo_loader(demo_json)
        else:
            self.qa_loader()

        self.coref = coreference(config, self.input)
        self.sentence_processor = sentence_processor(config, self.coref.output)
        self.to_stmt = to_statement(config, self.sentence_processor.output)
        self.output = self.to_stmt.output

    def demo_loader(self, demo_json):
        self.input.append(demo_json)
        return


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

