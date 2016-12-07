'''
Created on Nov, 2016

@author: hugo

'''

import sys
import os
import logging
import cPickle as pickle
import numpy as np
from learn2map import Learn2Map
from utils import *


if __name__ == '__main__':
    usage = 'python run_learn2map.py <path_to_pairs> <path_to_dict> <mapping_dim>'
    try:
        path_to_pairs = sys.argv[1]
        path_to_dict = sys.argv[2]
        mapping_dim = int(sys.argv[3])
    except:
        print usage
        sys.exit()

    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(sys.argv[0])
    logger.info("running %s" % " ".join(sys.argv))

    pairs = load_pickle(path_to_pairs)
    pairs = list(pairs)
    words = list(set([y for x in pairs for y in x]))
    vocab_dict = load_pickle(path_to_dict)

    # check if all words in pairs can be found in vocab_dict
    # if not, discard those words
    # from copy import deepcopy
    # tmp_pairs = deepcopy(pairs)
    # for a, b in tmp_pairs:
    #     if not a in vocab_dict or not b in vocab_dict:
    #         pairs.remove((a, b))

    # try:
    #     vocab_dict = get_emb2(sys.argv[2], set([item for sublist in pairs for item in sublist]))

    #     with open(os.path.splitext(path_to_pairs)[0]+'_dict.p', 'wb') as outfile:
    #         pickle.dump(vocab_dict, outfile)
    # except Exception as e:
    #     raise e



    # Generate training and test set
    train_ratio = 1.
    train_pairs = pairs[:int(len(pairs)*train_ratio)]

    test_pairs = pairs[int(len(pairs)*train_ratio):]
    # print len(test_pairs)
    # validation set
    np.random.seed(0)
    val_words = np.random.choice(words, int(len(words) * .3), replace=False)
    val_set = []
    for w in val_words:
        tmp = [y for x, y in pairs if x == w]
        if tmp:
            val_set.append((w, tmp))
    print 'val set size: %s' % len(val_set)

    try:
        # load the pretrained model
        l2m = Learn2Map().load_model(sys.argv[4])
        print 'loaded pretrained model.'
        print 'contine training.'
        # import pdb;pdb.set_trace()
        l2m = Learn2Map(dim=mapping_dim, alpha=1e-3, tol=1e-3, max_iter=10000, norm_ctr=1., verbose=2).fit(train_pairs, vocab_dict, val_set, C_init=l2m.C, L_init=l2m.L, save_per_iter=50, save_prefix='cont_')
    except:
        # Fit the model
        l2m = Learn2Map(dim=mapping_dim, alpha=1e-3, tol=1e-3, max_iter=10000, norm_ctr=1., verbose=2).fit(train_pairs, vocab_dict, val_set, save_per_iter=50)

    # save model
    try:
        l2m.save_model(sys.argv[4])
    except:
        l2m.save_model('biglearn2map.mod')


    # score = 0.
    # for x, y in val_set:
    #     pred = l2m.most_hypernyms(x, vocab_dict, topn=10)
    #     # score += jaccard_sim(y, pred)
    #     score += recall(y, pred)
    # score /= len(val_set)
    # print score
    # # test
    # print 'predicting hypernyms:'
    # for x, y in test_pairs[:10]:
    #     pred = l2m.most_hypernyms(x, vocab_dict, topn=10)
    #     print x
    #     print 'groundtruth: %s' % [b for a, b in pairs if a == x]
    #     print 'pred: %s' % pred
    #     sim = l2m.most_similar(x, vocab_dict, topn=10)
    #     print 'most similar: %s' % sim
    #     print
    # Predict
    # pred = l2m.predict(test_pairs, vocab_dict)
    # # print 'pred labels:'
    # # print pred
    # accuracy = calc_accuracy(pred, test_label)
    # print 's: %s' % accuracy
    # # import pdb;pdb.set_trace()

