import pickle, math
from nltk.corpus import wordnet as wn


true_label = 'SUBSTANCE'
labels = pickle.load( open( "labels.p", "rb" ) )
entities = list(labels[true_label])
not_found = []

n_entities = len(entities)
n_layers = 10

print ('true label of this cluster: %s' % true_label)
print ('total number of entities in this cluster: %d' % n_entities)
print ('max number of layers considered: %d' % n_layers)
print

entity_paths = {}
entity_hypernyms = {}

for entity in entities:
    entity_hypernyms[entity] = set()
    entity_paths[entity] = []
    synsets = wn.synsets(entity)
    if not synsets: 
        not_found.append(entity)
    for synset in synsets:
        paths = synset.hypernym_paths()
        for path in paths:
            path.reverse()
            entity_paths[entity].append(path)

print ('%d entities cannot be found in WordNet: ' % len(not_found))
print not_found
print

freqs = {}
distances = {}

for i in range(n_layers):
    print ('first %d layer(s): ' % (i + 1))
    for entity in entities:
        hypernyms = entity_hypernyms[entity]
        paths = entity_paths[entity]
        for path in paths:
            if i < len(path):
                curt = path[i]
                if not curt in hypernyms:
                    hypernyms.add(curt)
                    if not curt in freqs:
                        freqs[curt] = 0
                        distances[curt] = 0
                    freqs[curt] += 1
                    distances[curt] += (i + 1)

    metric = {}

    for cand in freqs.keys():
        metric[cand] = freqs[cand] * math.pow((freqs[cand] / float(distances[cand])), 0.75)

    sorted_metric = sorted(metric.items(), key=lambda d: d[1], reverse=True)
    print sorted_metric[0:10]
    print
    # if (sorted_freqs[0][1] >= n_entities * 0.3):
    #     break
