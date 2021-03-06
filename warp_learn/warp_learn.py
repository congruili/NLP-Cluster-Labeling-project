'''
Created on Nov, 2016

@author: hugo

This software implements the algorithm described in this paper:
[1] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Large scale image annotation: learning to rank with joint word-image embeddings." Machine learning 81.1 (2010): 21-35.

'''

import logging
from time import time
import numpy as np
from utils import *

class WarpLearn(object):
    """Warp Learn Model.

        A supervised cluster labeling model based on word embeddings.
        Pre-trained cluster and label embeddings are needed beforehand.

        Parameters
        ----------
        dim : dimensionality of hyper space.

        alpha : step size for SGD algorithm.

        tol : threshold for convergence.

        max_iter : maximum number of iterations.

        norm_ctr : norm constraint for mapping functions.

        verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

        verbose_interval : int, default to 10.
        Number of iteration done before the next print.
        """

    def __init__(self, dim=100, alpha=1e-2, tol=1e-5, max_iter=1000, norm_ctr=1.,
                verbose=0, verbose_interval=100):
        self.dim = dim
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.norm_ctr = norm_ctr
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self._check_initial_parameters()


    def fit(self, X, Y, Y_dict, eval_):
        """Estimate the mapping functions.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data array.

        Y : list, length n_samples
            The corresponding label for each data sample in X.

        Y_dict : dict
            The corresponding representation for all possible labels.

        eval_ : [array-like, list]
            evaluation set.

        Returns
        -------
        self
        """
        self._check_params(X, Y, Y_dict)

        n_samples, n_features = X.shape
        n_labels = len(Y_dict)

        # Initialization
        self._print_verbose_msg_init_beg()
        self.C = np.random.randn(self.dim, n_features) + 1. / np.sqrt(n_features)
        self.L = np.random.randn(self.dim, n_features) + 1. / np.sqrt(n_features)

        self.converged_ = False
        diff_CL = np.infty
        for n_iter in range(self.max_iter):
            prev_C = np.copy(self.C)
            prev_L = np.copy(self.L)
            change_ = False

            # Pick a random labeled example (xi, yi)
            pos_idx = np.random.choice(range(n_samples), 1)[0]
            pos_score = np.dot(np.dot(self.L, Y_dict[Y[pos_idx]]), np.dot(self.C, X[pos_idx]))

            labels = Y_dict.keys()
            labels.remove(Y[pos_idx])

            for N in range(1, n_labels):
                neg_label = np.random.choice(labels, 1)[0]
                neg_score = np.dot(np.dot(self.L, Y_dict[neg_label]), np.dot(self.C, X[pos_idx]))

                if neg_score > pos_score - 1:
                    break

            if neg_score > pos_score - 1:
                change_ = True
                k = int(np.floor((n_labels - 1.) / N))
                L_k = sum([1. / x for x in range(1, k + 1)])
                # Make a gradient step to minimize:
                # obj = L_k * (1. - pos_score + neg_score)
                diff_y = (Y_dict[neg_label] - Y_dict[Y[pos_idx]]).reshape((n_features, 1))
                f_c = L_k * np.dot(np.dot(self.L, diff_y), X[pos_idx].reshape((1, n_features)))
                f_l = L_k * np.dot(np.dot(self.C, X[pos_idx].reshape((n_features, 1))), diff_y.T)

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
                # diff_CL = np.linalg.norm(self.C - prev_C) + np.linalg.norm(self.L - prev_L)
                diff_CL = 1. - calc_accuracy(self.predict(eval_[0], Y_dict), eval_[1])

                if diff_CL < self.tol:
                    self.converged_ = True
                    break
            self._print_verbose_msg_iter_end(n_iter, diff_CL)

        self._print_verbose_msg_init_end(diff_CL)
        if not self.converged_:
            print 'Warning: Did not converged. Try different init parameters, or increase max_iter, tol or check for degenerate data.'

        return self

    def predict(self, X, Y_dict):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data array.

        Y_dict : dict
            The corresponding representation for all possible labels.

        Returns
        -------
        labels : list of array, shape (n_samples,)
        """
        self._check_features(X, Y_dict)
        self._check_is_fitted()
        labels = Y_dict.keys()
        Y = np.r_[Y_dict.values()]
        score = np.dot(np.dot(self.C, X.T).T, np.dot(self.L, Y.T))
        yidx = np.argmax(score, axis=1)

        return [labels[idx] for idx in yidx]

    def save_model(self, mod_file):
        try:
            with open(mod_file, 'wb') as mf:
                np.savez(mf, C=self.C, L=self.L)
        except Exception as e:
            raise e
        else:
            mf.close()

        return self

    def load_model(self, mod_file):
        try:
            with open(mod_file, 'r') as mf:
                npzfile = np.load(mf)
                self.C = npzfile['C']
                self.L = npzfile['L']
        except Exception as e:
            raise e
        else:
            mf.close()

        return self

    def _check_initial_parameters(self):
        """Check values of the basic parameters.
        """
        if not isinstance(self.dim, int) or self.dim < 1:
            raise ValueError("Invalid value for 'dim': %s " % self.dim)
        if not isinstance(self.alpha, float) or self.alpha <= 0:
            raise ValueError("Invalid value for 'alpha': %s " % self.alpha)
        if not isinstance(self.tol, (int, float)) or self.tol <= 0:
            raise ValueError("Invalid value for 'tol': %s " % self.tol)
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %s " % self.max_iter)
        if not isinstance(self.norm_ctr, (int, float)) or self.norm_ctr < 1:
            raise ValueError("Invalid value for 'norm_ctr': %s " % self.norm_ctr)
        if not isinstance(self.verbose, int) or self.verbose < 0:
            raise ValueError("Invalid value for 'verbose': %s " % self.verbose)
        if not isinstance(self.verbose_interval, int) or self.verbose_interval < 1:
            raise ValueError("Invalid value for 'verbose_interval': %s " % self.verbose_interval)

    def _check_params(self, X, Y, Y_dict):
        """Check values of the parameters.
        """
        self._check_samples(X, Y)
        self._check_features(X, Y_dict)

    def _check_samples(self, X, Y):
        if X.shape[0] != len(Y):
            raise ValueError("The parameter X and Y should have consistent rows, "
                    "but got X: %s and Y: %s" % (X.shape[0], len(Y)))

    def _check_features(self, X, Y_dict):
        if X.shape[1] != Y_dict.values()[0].shape[0]:
            raise ValueError("X and Y_dict should have same number of features, "
                    "but got X: %s and Y_dict: %s" % (X.shape[1], Y_dict.values()[0].shape[0]))

    def _check_is_fitted(self):
        """Check if the model is fitted.
        """
        check_is_fitted(self, ['C', 'L'])

    def _print_verbose_msg_init_beg(self):
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization")
        elif self.verbose >= 2:
            print("Initialization")
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time

    def _print_verbose_msg_iter_end(self, n_iter, diff_CL, val_score=None):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("  Iteration %d" % n_iter)
            elif self.verbose >= 2:
                cur_time = time()
                print("  Iteration %d\t time lapse %.5fs\t diff_CL %.5f%s" % (
                    n_iter, cur_time - self._iter_prev_time, diff_CL, '\t val_score %s' % val_score if val_score else ''))
                self._iter_prev_time = cur_time

    def _print_verbose_msg_init_end(self, diff_CL):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Initialization converged: %s" % self.converged_)
        elif self.verbose >= 2:
            print("Initialization converged: %s\t time lapse %.5fs\t diff_CL %.5f" %
                  (self.converged_, time() - self._init_prev_time, diff_CL))
