import pickle, gensim
from nltk.corpus import wordnet as wn
import sys


def get_pairs(anchor):
    pairs = set()
    synsets = wn.synsets(anchor)
    if synsets:
        for synset in synsets:
            paths = synset.hypernym_paths()
            for path in paths:
                path.reverse()
                # path = path[:-4]
                path = path[:-1]
                if len(path) >= 2:
                    if model.vocab.has_key(anchor):
                        new_path = [anchor]
                    else:
                        new_path = []
                    # new_path = []
                    for node in path:
                        # node = str(node)[8:-2]
                        node = node.name().lower()
                        new_node = node[:(node.find('.'))]
                        if model.vocab.has_key(new_node):
                            if not new_node in new_path:
                                new_path.append(new_node)

                    if len(new_path) >= 2:
                        length = len(new_path)
                        for i in xrange(length - 1):
                            for j in xrange(i + 1, length):
                                pair = (new_path[i], new_path[j])
                                pairs.add(pair)

    return pairs


if __name__ == '__main__':

    print "running %s" % " ".join(sys.argv)
    model = gensim.models.Word2Vec.load_word2vec_format(sys.argv[1], binary=True)
    # model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    labels = pickle.load( open( sys.argv[2], "rb" ) )
    # labels = pickle.load( open( "labels.p", "rb" ) )

    label_words = set([y.lower() for x in labels.keys() for y in x.split('/')])
    if '' in label_words:
        label_words.remove('')

    anchors = set([y.lower() for x in labels.values() for y in x])
    anchors.update(label_words)
    print '# of anchors: %s' % len(anchors)

    all_pairs = set()
    for each in anchors:
        all_pairs.update(get_pairs(each))

    print '# of pairs: %s' % len(all_pairs)
    print 'vocab size: %s' % len(set([y for x in all_pairs for y in x]))
    pickle.dump( all_pairs, open( "bigpairs.p", "wb+" ) )
    import pdb;pdb.set_trace()

