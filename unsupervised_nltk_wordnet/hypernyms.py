import pickle
from nltk.corpus import wordnet as wn

labels = pickle.load( open( "labels.p", "rb" ) )
entities = labels['LANGUAGE']
freqs = {}

for entity in entities:
    synsets = wn.synsets(entity)
    size = len(synsets)
    if (size > 5):
        synsets = synsets[0:5]
    for synset in synsets:
        for hypernym in synset.hypernyms():
            if not hypernym in freqs:
                freqs[hypernym] = 0
            freqs[hypernym] += 1

sorted_freqs = sorted(freqs.items(), key=lambda d: d[1], reverse=True)
print len(entities)
print sorted_freqs
