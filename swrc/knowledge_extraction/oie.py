from openie import StanfordOpenIE

def stanfordOIE(texts):
    with StanfordOpenIE() as client:
        result = []
        for text in texts:
            result.append(client.annotate(text))
        return result

class oie:
    def __init__(self, config, input):
        self.input = input
        self.config = config
        self.output = self.run()

    def run(self):
        with StanfordOpenIE() as client:
            if self.config['mode'] == 'qa':
                for qa in self.input:
                    for utter in qa['utterances']:
                        for sent in utter['sents']:
                            if self.config['extraction']['oie'] == 'None':
                                sent['triples'] = []
                            else:
                                sent['triples'] = client.annotate(sent['statement'])
            elif self.config['mode'] == 'subtitle' or self.config['mode'] == 'demo':
                for ep in self.input:
                    for scene in ep:
                        for u in scene['scene']:
                            for sent in u['sents']:
                                if self.config['extraction']['oie'] == 'None':
                                    sent['triples'] = []
                                else:
                                    sent['triples'] = client.annotate(sent['statement'])

            print('Stanford Open IE done..')

        return self.input
