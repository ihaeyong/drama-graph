class to_statement:
    def __init__(self, config, input):
        self.config = config
        self.input = input

        self.output = self.statementization()

    def statementization(self):
        input = self.input
        if self.config['mode'] == 'qa':
            for qa in input:
                for u in qa['utterances']:
                    for sent in u['sents']:
                        sent['statement'] = sent['text']
        return input