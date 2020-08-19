from openie import StanfordOpenIE
from stanfordcorenlp import StanfordCoreNLP
import json
def stanfordOIE(texts):
    with StanfordOpenIE() as client:
        result = []
        for text in texts:
            result.append(client.annotate(text))
        return result

class oie:
    def __init__(self, config, input, nlp_parser):
        self.input = input
        self.config = config
        self.nlp_parser = nlp_parser
        self.output = self.run()

    def run(self):
        if self.nlp_parser == None:
            if self.config['mode'] == 'demo':
                return self.input
            self.nlp_parser = StanfordCoreNLP('data/stanford-corenlp-4.0.0')

        for ep in self.input:
            for scene in ep:
                for u in scene['scene']:
                    for sent in u['sents']:
                        if self.config['extraction']['oie'] == 'None':
                            sent['triples'] = []
                        else:
                            output = self.nlp_parser.annotate(sent['statement'], properties={
                                'annotators': 'openie',
                                'outputFormat': 'json'
                            })

                            output = json.loads(output)
                            sent['triples'] = []
                            for s in output['sentences']:
                                for result in s['openie']:
                                    del result['subjectSpan']
                                    del result['relationSpan']
                                    del result['objectSpan']
                                sent['triples'] += s['openie']

                            # sent['triples'] = client.annotate(sent['statement'])

            print('Stanford Open IE done..')

        return self.input
