import pickle
from nltk.corpus import wordnet as wn

labels = pickle.load( open( "labels.p", "rb" ) )
entities = list(labels['DISEASE'])
# print  entities
freqs = {}
list_of_pairs = [(entities[e1], entities[e2]) for e1 in range(len(entities)) for e2 in range(e1+1,len(entities))]

for e1, e2 in list_of_pairs:
    synsets1 = wn.synsets(e1)
    synsets2 = wn.synsets(e2)
    for synset1 in synsets1:
        for synset2 in synsets2:
            lchs = synset1.lowest_common_hypernyms(synset2)
            for lch in lchs:
                if not lch in freqs:
                    freqs[lch] = 0
                freqs[lch] += 1

sorted_freqs = sorted(freqs.items(), key=lambda d: d[1], reverse=True)
# print len(entities)
print sorted_freqs[0:10]

cands = []
for key, val in sorted_freqs[0:10]:
    cands.append(key)

similarities = {}

for cand in cands:
    val = 0.0
    time = 0
    for entity in entities:
        synsets = wn.synsets(entity)
        for synset in synsets:
            curt = wn.wup_similarity(synset, cand)
            if curt:
                val += curt
                time += 1
    similarities[cand] = val / time

sorted_similarities = sorted(similarities.items(), key=lambda d: d[1], reverse=True)

print sorted_similarities



