import spacy

class sentence_processor:
    def __init__(self, config, input):
        self.input = input
        self.config = config
        self.output = self.segmentation()

    def sentence_typing(self, sent):
        '''
                statement:      subject + verb .. + '.'
                question:       auxiliary verb + subject + verb .. + '?'
                command:        verb + .. + '.' or '!'
                exclamation:    What or How + .. + '!'
        '''
        tokenized = [(tok.text, tok.pos_) for tok in sent]
        if tokenized[0][0].lower() in ['what', 'how'] and tokenized[-1][0] == '!':
            return 'exclamation'
        if tokenized[-1][0] == '?' and tokenized[0][1] == 'AUX':
            return 'question'
        if tokenized[0][1] == 'VERB':
            return 'command'
        return 'statement'

    def segmentation(self):
        nlp = spacy.load('en_core_web_sm')
        json = self.input
        if self.config['mode'] == 'qa':
            for qa in json:
                for utter in qa['utterances']:
                    text = utter['utter']
                    sents = nlp(text).sents
                    utter['sents'] = []
                    for s in sents:
                        s_type = self.sentence_typing(s)
                        t = {'type': s_type, 'origin': s.text}
                        utter['sents'].append(t)
        elif self.config['mode'] == 'subtitle':
            for j in self.input:
                for scene in j:
                    for utter in scene['scene']:
                        text = utter['utter']
                        sents = nlp(text).sents
                        utter['sents'] = []
                        for s in sents:
                            s_type = self.sentence_typing(s)
                            t = {'type': s_type, 'origin': s.text}
                            print(t)
                            utter['sents'].append(t)
        return json




