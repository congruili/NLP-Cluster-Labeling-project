'''
Created on Nov, 2016

@author: hugo

'''

import sys
import cPickle as pickle
from collections import Counter, defaultdict
from learn2map import Learn2Map
from utils import *


def pred_clus(entities, model, vocab_dict, factor=1., cands=None, topn=5, method=0):
    labels = defaultdict(list)
    scores = {}
    i = 0.
    for each in entities:
        each = '_'.join(each.split()).lower()
        if not each in vocab_dict:
            i += 1
            continue
        if method == 0:
            hypernyms, sim = model.most_hypernyms_by_cands(each, vocab_dict, topn=topn, cands=cands)
        else:
            hypernyms, sim = model.most_similar_by_cands(each, vocab_dict, topn=topn, cands=cands)
        for idx in range(len(hypernyms)):
            # key is the candidate
            # value is a list of all of its rankings for each entity if exists
            # labels[hypernyms[idx]].append(sim[idx])
            labels[hypernyms[idx]].append(idx)

    # if (1 - i / len(entities)) < .7:
        # return None
    print 'hit ratio: %s' % (1 - i / len(entities))
    for k, v in labels.items():
        # scores[k] = sum([x**factor for x in v])
        scores[k] = sum([1. / (x + 1)**factor for x in v])
    return zip(*sorted(scores.items(), key=lambda d:d[1], reverse=True))


if __name__ == '__main__':
    usage = 'python clusl.py <path_to_clus> <path_to_dict> <mod_file> <label_cands>'
    try:
        path_to_clus = sys.argv[1]
        path_to_dict = sys.argv[2]
        mod_file = sys.argv[3]
    except:
        print usage
        sys.exit()

    clus  = load_pickle(path_to_clus)
    vocab_dict = load_pickle(path_to_dict)
    if len(sys.argv) == 5:
        label_cands  = load_pickle(sys.argv[4])
    else:
        label_cands = None


    # vocab_dict = get_emb2(sys.argv[4], set([y for x in clus.values() for y in x]))
    # vocab_dict2 = load_pickle(path_to_dict)
    # vocab_dict.update(vocab_dict2)
    l2m = Learn2Map().load_model(mod_file)


    import numpy as np
    # label_set = [x for x in np.random.choice(clus.keys(), 40, replace=False) if len(clus[x]) > 10]
    # label_set = clus.keys()

    # import pdb;pdb.set_trace()
    # label_set = ['FACILITY:AIRPORT', 'ORGANIZATION:HOSPITAL', 'ORGANIZATION:RELIGIOUS', 'ORGANIZATION:HOTEL']
    label_set = ['SUBSTANCE:CHEMICAL', 'PRODUCT:WEAPON', 'LOCATION:REGION']
    results = {}
    for label in label_set:
        entities = clus[label]
        preds = pred_clus(entities, l2m, vocab_dict, factor=3., \
                cands=list(label_cands[label]) if label_cands != None and label in label_cands else None, topn=10, method=0)
        if preds:
            # print 'entities: %s' % entities
            print 'groundtruth: %s' % label.lower()
            print 'preds: %s' % list(preds[0])[:10]
            print
            results[label] = list(preds[0])[:10]
    pickle.dump(results, open('results_sim.p','w'))
    import pdb;pdb.set_trace()
