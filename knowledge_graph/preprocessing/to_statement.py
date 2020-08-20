
class to_statement:
    def __init__(self, config, input):
        self.config = config
        self.input = input

        self.output = self.statementization()

    def statementization(self):
        input = self.input
        if self.config['mode'] == 'subtitle' or self.config['mode'] == 'demo':
            for ep in input:
                for scene in ep:
                    for u in scene['scene']:
                        for sent in u['sents']:
                            sent['statement'] = sent['origin']

        return input