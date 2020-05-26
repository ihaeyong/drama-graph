from openie import StanfordOpenIE

def stanfordOIE(texts):
    with StanfordOpenIE() as client:
        result = []
        for text in texts:
            result.append(client.annotate(text))
        return result

class oie:
    def __init__(self, input, type):
        self.input = input
        self.type = type
        self.output = self.run()

    def run(self):
        if self.type == 'stanford':
            with StanfordOpenIE() as client:
                print()

                for qa in self.input:
                    for utter in qa['utterances']:
                        for sent in utter['sents']:
                            sent['triples'] = client.annotate(sent['text'])

        return self.input
