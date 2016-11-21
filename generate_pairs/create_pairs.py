import pickle, gensim
from nltk.corpus import wordnet as wn

model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
labels = pickle.load( open( "labels.p", "rb" ) )
pairs = set()

for label in labels.keys():
    entities = labels[label]
    for entity in entities:
        synsets = wn.synsets(entity)
        if synsets:
            for synset in synsets:
                paths = synset.hypernym_paths()
                for path in paths:
                    path.reverse()
                    path = path[:-4]
                    if len(path) >= 2:
                        new_path = []
                        for node in path:
                            node = str(node)[8:-2]
                            new_node = node[:(node.find('.'))]
                            if model.vocab.has_key(new_node):
                                new_path.append(new_node)

                        if len(new_path) >= 2:
                            length = len(new_path)
                            for i in xrange(length - 1):
                                for j in xrange(i + 1, length):
                                    pair = (new_path[i], new_path[j])
                                    pairs.add(pair)

pickle.dump( pairs, open( "pairs.p", "wb+" ) )
print pairs
print len(pairs)
