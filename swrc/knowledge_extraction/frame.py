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
            for qa in self.input:
                for utter in qa['utterances']:
                    for sent in utter['sents']:
                        sent['frames'] = eng_frameBERT(sent['statement'])
            print('frameBERT done..')

        return self.input
