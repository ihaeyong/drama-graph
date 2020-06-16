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
                        sent['statement'] = sent['origin']
        elif self.config['mode'] == 'subtitle':
            print()
            for ep in input:
                for scene in ep:
                    for u in scene['scene']:
                        for sent in u['sents']:
                            sent['statement'] = sent['origin']
        return input