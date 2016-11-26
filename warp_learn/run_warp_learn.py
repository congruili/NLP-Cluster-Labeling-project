'''
Created on Nov, 2016

@author: hugo

'''

import sys
import logging
import cPickle as pickle
import numpy as np
from warp_learn import WarpLearn
from utils import *


if __name__ == '__main__':
    usage = 'python run.py <path_to_clus> <path_to_label> <mapping_dim>'
    try:
        path_to_clus = sys.argv[1]
        path_to_label = sys.argv[2]
        mapping_dim = int(sys.argv[3])
    except:
        print usage
        sys.exit()

    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(sys.argv[0])
    logger.info("running %s" % " ".join(sys.argv))


    label_dict = load_pickle(path_to_label)
    # remove clusters with unknown labels
    clus = load_pickle(path_to_clus)
    for k in clus.keys():
        if not k in label_dict:
            del clus[k]

    # label_dict = get_emb(sys.argv[4], label_dict.keys())
    # with open("label_dict.p", 'wb') as outfile:
    #     pickle.dump(label_dict, outfile)

    # Generate training and test set
    labels = clus.keys()
    clusters = np.r_[clus.values()]
    train_ratio = .98

    train_label = labels[:int(len(labels)*train_ratio)]
    train_clus = clusters[:int(clusters.shape[0]*train_ratio)]

    test_label = labels[int(len(labels)*train_ratio):]
    test_clus = clusters[int(clusters.shape[0]*train_ratio):]
    print len(test_clus)
    # Fit the model
    wl = WarpLearn(dim=mapping_dim, alpha=1e-4, tol=1e-3, max_iter=1000, norm_ctr=2., verbose=2).fit(train_clus, train_label, label_dict, [test_clus, test_label])

    # save model
    # wl = WarpLearn(dim=mapping_dim, alpha=1e-4, tol=1e-3, max_iter=1000, norm_ctr=2., verbose=2).load_model('warp.mod')

    # Predict
    pred = wl.predict(test_clus, label_dict)
    # print 'pred labels:'
    # print pred
    accuracy = calc_accuracy(pred, test_label)
    print 's: %s' % accuracy

    # save model
    # wl.save_model('warp.mod')

