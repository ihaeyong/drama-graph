from openie import StanfordOpenIE

text = 'I love you.'
with StanfordOpenIE() as client:
    result = client.annotate(text)
    print()


