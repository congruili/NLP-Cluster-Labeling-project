'''
Created on Nov, 2016

@author: hugo

'''

import sys
import cPickle as pickle
from collections import Counter
from learn2map import Learn2Map
from utils import *


def pred_clus(entities, model, vocab_dict):
    labels = []
    i = 0.
    for each in entities:
        each = '_'.join(each.split()).lower()
        if not each in vocab_dict:
            i += 1
            continue
        hypernyms = model.most_hypernyms(each, vocab_dict, topn=5)
        labels.extend(hypernyms)
    if 1 - i / len(entities) > .7:
        print 'hit ratio: %s' % (1 - i / len(entities))
        return zip(*sorted(Counter(labels).items(), key=lambda d:d[1], reverse=True))
    else:
        return None


if __name__ == '__main__':
    usage = 'python clusl.py <path_to_clus> <path_to_dict> <mod_file>'
    try:
        path_to_clus = sys.argv[1]
        path_to_dict = sys.argv[2]
        mod_file = sys.argv[3]
    except:
        print usage
        sys.exit()

    clus  = load_pickle(path_to_clus)
    vocab_dict = load_pickle(path_to_dict)
    # vocab_dict = get_emb2(sys.argv[4], set([y for x in clus.values() for y in x]))
    # vocab_dict2 = load_pickle(path_to_dict)
    # vocab_dict.update(vocab_dict2)
    l2m = Learn2Map().load_model(mod_file)

    for label, entities in clus.items()[:20]:
        if label.lower() != 'substance:drug':
            continue
        preds = pred_clus(entities, l2m, vocab_dict)
        # import pdb;pdb.set_trace()
        if preds:
            print 'entities: %s' % entities
            print 'groundtruth: %s' % label.lower()
            print 'preds: %s' % list(preds[0])
            print
