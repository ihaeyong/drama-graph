import requests

def eng_frameBERT(text):
    host = 'http://143.248.135.188:1106/frameBERT'
    data = {
        'text': text
    }
    response = requests.post(host, data=data, verify=False)
    return response.json()

class frame:
    def __init__(self, config, input):
        self.input = input
        self.config = config
        self.output = self.run()

    def run(self):
        if self.config['extraction']['frame'] == 'frameBERT':
            if self.config['mode'] == 'qa':
                for qa in self.input:
                    for utter in qa['utterances']:
                        for sent in utter['sents']:
                            sent['frames'] = eng_frameBERT(sent['statement'])
            elif self.config['mode'] == 'subtitle':
                for ep in self.input:
                    for scene in ep:
                        for u in scene['scene']:
                            for sent in u['sents']:
                                sent['frames'] = []
                                results = eng_frameBERT(sent['statement'])
                                if type(results) is dict:  # error case
                                    continue

                                cur = None
                                for element in results:
                                    if element[1][-2:] == 'lu':
                                        if cur:
                                            sent['frames'].append(cur)
                                            cur = None
                                        cur = {'frame':element[0].split(':')[-1], 'lu': element[2]}
                                    else:
                                        cur[element[1].split('-')[-1]] = element[2]
                                if cur:
                                    sent['frames'].append(cur)
            print('frameBERT done..')

        return self.input
