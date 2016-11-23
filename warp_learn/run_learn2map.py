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
    usage = 'python run_learn2map.py [path_to_pairs] [path_to_dict] [mapping_dim]'
    try:
        path_to_pairs = sys.argv[1]
        path_to_dict = sys.argv[2]
        mapping_dim = int(sys.argv[3])
    except:
        print usage

    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(sys.argv[0])
    logger.info("running %s" % " ".join(sys.argv))

    pairs = load_pickle(path_to_pairs)
    vocab_dict = load_pickle(path_to_dict)
    # try:
    #     vocab_dict = get_emb2(sys.argv[3], set([item for sublist in pairs for item in sublist]))

    #     with open(os.path.splitext(path_to_pairs)[0]+'_dict.p', 'wb') as outfile:
    #         pickle.dump(vocab_dict, outfile)
    # except Exception as e:
    #     raise e

    # Generate training and test set
    train_ratio = .8
    pairs = list(pairs)
    train_pairs = pairs[:int(len(pairs)*train_ratio)]

    test_pairs = pairs[int(len(pairs)*train_ratio):]
    # print len(test_pairs)

    try:
        # load the pretrained model
        l2m = Learn2Map().load_model(sys.argv[4])
        print 'loaded pretrained model.'
    except:
        # Fit the model
        l2m = Learn2Map(dim=mapping_dim, alpha=1e-3, tol=1e-3, max_iter=1000, norm_ctr=2., verbose=2).fit(train_pairs, vocab_dict, test_pairs)

    # # save model
    # try:
    #     l2m.save_model(sys.argv[4])
    # except:
    #     l2m.save_model('learn2map.mod')

    # test
    print 'predicting hypernyms:'
    print 'token - groundtruth - pred'
    for x, y in test_pairs[:10]:
        pred = l2m.most_hypernyms(x, vocab_dict, topn=10)
        print "%s - %s - %s" % (x, y, pred)
        sim = l2m.most_similar(x, vocab_dict, topn=10)
        print "%s" % sim
        print
    # Predict
    # pred = l2m.predict(test_pairs, vocab_dict)
    # # print 'pred labels:'
    # # print pred
    # accuracy = calc_accuracy(pred, test_label)
    # print 's: %s' % accuracy
    # # import pdb;pdb.set_trace()

