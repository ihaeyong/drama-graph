import requests

def eng_frameBERT(text):
    host = 'http://143.248.135.188:1106/frameBERT'
    data = {
        'text': text
    }
    response = requests.post(host, data=data, verify=False)
    return response.json()

class frame:
    def __init__(self, input, type):
        self.input = input
        self.type = type
        self.output = self.run()

    def run(self):
        if self.type == 'frameBERT':
                for qa in self.input:
                    for utter in qa['utterances']:
                        for sent in utter['sents']:
                            sent['frames'] = eng_frameBERT(sent['text'])

        return self.input
