import spacy

nlp = spacy.load('en_core_web_sm')

# text = r'Hurry and eat already. Huh? She\'ll ask how it was, so you should at least taste it.'
text = r'Stop!'
sents = nlp(text).sents

for s in sents:
    tokenized = [(tok.text, tok.pos_) for tok in s]
    print(tokenized)




# import re
#
# coref_p = re.compile(r'[(].*[)]')
# sent = r'Is it her(Haeyoung1)? Is the girl, remember, he(Dokyung) hugged and twirled in front of us(Yijoon, Hun, Sangseok, Gitae)?'
#
# open = []
# close = []
# patts = []
# corefs = []
#
# for i, char in enumerate(sent):
#     if char == '(':
#         open.append(i)
#     if char == ')':
#         close.append(i)
#
# if len(open) != len(close):
#     print('error')
#
#
#
# for i in range(len(open)):
#     patt = sent[open[i]:close[i]+1]
#     patts.append(patt)
#     mention = sent[:open[i]].split()[-1]
#     st = open[i] - len(mention)
#     en = open[i]
#     print(patt)
#     print(sent[st:en])
#
#     coref = {
#         'begin': st,
#         'end': en,
#         'form': sent[st:en],
#         'coref': patt[1:-1]
#     }
#
#     corefs.append(coref)
#
#
# new_sent = sent
# idx = 0
# for ii, patt in enumerate(patts):
#     new_sent = new_sent.replace(patt, '')
#     corefs[ii]['begin'] -= idx
#     corefs[ii]['end'] -= idx
#     idx += len(patt)
#
# print(new_sent)
# print(corefs)
#
