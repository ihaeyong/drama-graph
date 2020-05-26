import spacy

def to_statement(text):
    s_type = ''
    statement = text



    return s_type, statement

class sentence_divider:
    def __init__(self, input):
        self.input = input
        self.output = self.segmentation()

    def segmentation(self):
        nlp = spacy.load('en')
        json = self.input

        for qa in json:
            for utter in qa['utterances']:
                text = utter['utter']
                sents = nlp(text).sents
                utter['sents'] = []
                for s in sents:
                    s_type, statement = to_statement(s.text)
                    t = {'origin': s.text, 'text': statement, 'type': s_type}
                    utter['sents'].append(t)
                    print(s.text)
        return json




