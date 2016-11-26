'''
Created on Nov, 2016

@author: hugo

'''

import numpy as np
from warp_learn import WarpLearn
from utils import *

class Learn2Map(WarpLearn):
    """Learn2Map Model.

    Learn to map hypernym relations in word embedding space.
    """

    def __init__(self, dim=100, alpha=1e-2, tol=1e-5, max_iter=1000, norm_ctr=1.,
                verbose=0, verbose_interval=100):
        super(Learn2Map, self).__init__(dim=dim, alpha=alpha, tol=tol,
                max_iter=max_iter, norm_ctr=norm_ctr, verbose=verbose,
                verbose_interval=verbose_interval)

    def fit(self, pairs, vocab_dict, val_set):
        """Estimate the mapping functions.

        Parameters
        ----------
        pairs : list of tuples
            The hyponym-hypernym pairs.

        vocab_dict : dict
            Word embeddings.

        val_set :
            validation set.

        Returns
        -------
        self
        """
        # self._check_params(X, Y, Y_dict)

        # n_samples, n_features = X.shape
        margin = 3.
        n_samples = len(pairs)
        n_features = vocab_dict.values()[0].shape[0]
        n_vocabs = len(vocab_dict)

        # Initialization
        self._print_verbose_msg_init_beg()
        self.C = np.random.randn(self.dim, n_features) + 1. / np.sqrt(n_features)
        self.L = np.random.randn(self.dim, n_features) + 1. / np.sqrt(n_features)
        # import pdb;pdb.set_trace()
        self.converged_ = False
        diff_CL = np.infty
        for n_iter in range(self.max_iter):
            prev_C = np.copy(self.C)
            prev_L = np.copy(self.L)
            change_ = False

            # Pick a random labeled example (xi, yi)
            pos_idx = np.random.choice(range(n_samples), 1)[0]
            pos_score = np.dot(np.dot(self.L, vocab_dict[pairs[pos_idx][1]]), np.dot(self.C, vocab_dict[pairs[pos_idx][0]]))

            vocabs = set(vocab_dict.keys())
            filter_ = set([y for x, y in pairs if x == pairs[pos_idx][0]])
            filter_.add(pairs[pos_idx][0])
            vocabs -= filter_

            for N in range(1, len(vocabs) + 1):
                neg_label = np.random.choice(list(vocabs), 1)[0]
                neg_score = np.dot(np.dot(self.L, vocab_dict[neg_label]), np.dot(self.C, vocab_dict[pairs[pos_idx][0]]))

                if neg_score > pos_score - margin:
                    break

            if neg_score > pos_score - margin:
                change_ = True
                k = int(np.floor((len(vocabs)) / N))
                L_k = sum([1. / x for x in range(1, k + 1)])
                # Make a gradient step to minimize:
                # obj = L_k * (1. - pos_score + neg_score)
                diff_y = (vocab_dict[neg_label] - vocab_dict[pairs[pos_idx][1]]).reshape((n_features, 1))
                f_c = L_k * np.dot(np.dot(self.L, diff_y), vocab_dict[pairs[pos_idx][0]].reshape((1, n_features)))
                f_l = L_k * np.dot(np.dot(self.C, vocab_dict[pairs[pos_idx][0]].reshape((n_features, 1))), diff_y.T)

                self.C -=  self.alpha * f_c
                self.L -=  self.alpha * f_l

                # Project weights to enforce constraints
                # ||Ci|| <= self.norm_ctr (1)
                # ||Li|| <= self.norm_ctr (2)
                for idx in range(n_features):
                    C_norm = np.linalg.norm(self.C[:, idx])
                    L_norm = np.linalg.norm(self.L[:, idx])
                    if C_norm > self.norm_ctr:
                        self.C[:, idx] = self.C[:, idx] / C_norm * self.norm_ctr
                    if L_norm > self.norm_ctr:
                        self.L[:, idx] = self.L[:, idx] / L_norm * self.norm_ctr

            # In original paper, convergence conditino is validation error does not improve
            # Here, we use a different strategy
            if change_:
                diff_CL = np.linalg.norm(self.C - prev_C) + np.linalg.norm(self.L - prev_L)

                # calc jaccard sim
                if n_iter % self.verbose_interval == 0:
                    score = 0.
                    for x, y in val_set:
                        pred = self.most_hypernyms(x, vocab_dict, topn=10)
                        # score += jaccard_sim(y[:10], pred)
                        score += recall(y, pred)
                    score /= len(val_set)
                # diff_CL = 1. - calc_accuracy(self.predict(eval_[0], Y_dict), eval_[1])
                    print score
                if diff_CL < self.tol:
                    self.converged_ = True
                    break
            self._print_verbose_msg_iter_end(n_iter, diff_CL, None)

        self._print_verbose_msg_init_end(diff_CL)
        if not self.converged_:
            print 'Warning: Did not converged. Try different init parameters, or increase max_iter, tol or check for degenerate data.'

        return self

    def most_hypernyms(self, token, vocab_dict, topn=5):
        if not token in vocab_dict:
            raise ValueError("The token `%s` is not in vocab_dict." % token)
        self._check_is_fitted()
        vocabs = vocab_dict.keys()
        Y = np.r_[vocab_dict.values()]
        X = vocab_dict[token]
        score = np.dot(np.dot(self.C, X.T).T, np.dot(self.L, Y.T))
        yidx = score.argsort()[::-1][:topn]

        return [vocabs[idx] for idx in yidx]

    def most_similar(self, token, vocab_dict, topn=5):
        """cosine similarity"""
        if not token in vocab_dict:
            raise ValueError("The token `%s` is not in vocab_dict." % token)

        vocabs = vocab_dict.keys()
        Y = np.r_[vocab_dict.values()]
        X = vocab_dict[token]
        score = np.dot(unitvec(X), unitmatrix(Y).T)
        score[vocabs.index(token)] = 0 # ignore the token itself
        yidx = score.argsort()[::-1][:topn]

        return [vocabs[idx] for idx in yidx]
