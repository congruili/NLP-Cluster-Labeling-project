import pickle
from nltk.corpus import wordnet as wn

labels = pickle.load( open( "labels.p", "rb" ) )
entities = list(labels['GPE:CITY'])

n_entities = len(entities)
n_layers = 10

levels = []
for i in range(n_layers):
    levels.append([])

for entity in entities:
    synsets = wn.synsets(entity)
    for synset in synsets:
        paths = synset.hypernym_paths()
        for path in paths:
            path.reverse()
            for i in range(min(len(path), n_layers)):
                levels[i].append(path[i])

freqs = {}
for i in range(n_layers):
    for hypernym in levels[i]:
        if not hypernym in freqs:
            freqs[hypernym] = 0
        freqs[hypernym] += 1

    sorted_freqs = sorted(freqs.items(), key=lambda d: d[1], reverse=True)
    print sorted_freqs[0:10]
    print
    if (sorted_freqs[0][1] > n_entities * 0.8):
        break


