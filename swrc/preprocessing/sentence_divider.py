import spacy

class sentence_divider:
    def __init__(self, config, input):
        self.input = input
        self.config = config
        self.output = self.segmentation()

    def segmentation(self):
        nlp = spacy.load('en')
        json = self.input
        if self.config['mode'] == 'qa':
            for qa in json:
                for utter in qa['utterances']:
                    text = utter['utter']
                    sents = nlp(text).sents
                    utter['sents'] = []
                    for s in sents:
                        t = {'origin': s.text}
                        utter['sents'].append(t)
                        print(s.text)
        elif self.config['mode'] == 'subtitle':
            for j in self.input:
                for scene in j:
                    for utter in scene['scene']:
                        print()
                        text = utter['utter']
                        sents = nlp(text).sents
                        utter['sents'] = []
                        for s in sents:
                            t = {'origin': s.text}
                            utter['sents'].append(t)
                            print(s.text)

        return json




