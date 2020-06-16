from utils.macro import *

input = 'data/output/extracted.json'
output = 'data/output/simple_extracted.json'

j = jsonload(input)

for qa in j:
    for u in qa['utterances']:


        del u['old_utter']
        del u['triples']
        del u['frameBERT']
        for s in u['sents']:
            del s['origin']
            del s['statement']

            coref_dict = {}
            for cr in s['corefs']:
                coref_dict[cr['form']] = cr['coref']
            del s['corefs']

            frames = s['frames']
            del s['frames']
            frame_dict = {}

            for t in s['triples']:
                if t['subject'] in coref_dict:
                    t['subject'] = coref_dict[t['subject']]
                if t['object'] in coref_dict:
                    t['object'] = coref_dict[t['object']]

            for f in frames:
                if f[0] not in frame_dict:
                    frame_dict[f[0]] = []
                if f[2] in coref_dict:
                    f[2] = coref_dict[f[2]]
                frame_dict[f[0]].append((f[1],f[2]))
            s['frames'] = frame_dict



jsondump(j, output)
