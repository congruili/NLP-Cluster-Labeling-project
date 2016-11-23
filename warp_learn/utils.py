'''
Created on Nov, 2016

@author: hugo

'''

import sys
import numpy as np
from scipy import sparse
import cPickle as pickle

def unitmatrix(matrix, norm='l2', axis=1):
    if norm == 'l1':
        maxtrixlen = np.sum(np.abs(matrix), axis=axis)
    if norm == 'l2':
        maxtrixlen = np.linalg.norm(matrix, axis=axis)

    if np.any(maxtrixlen <= 0):
        return matrix
    else:
        maxtrixlen = maxtrixlen.reshape(1, len(maxtrixlen)) if axis == 0 else maxtrixlen.reshape(len(maxtrixlen), 1)
        return matrix / maxtrixlen


def unitvec(vec, norm='l2'):
    """
    Scale a vector to unit length. The only exception is the zero vector, which
    is returned back unchanged.
    Output will be in the same format as input (i.e., gensim vector=>gensim vector,
    or np array=>np array, scipy.sparse=>scipy.sparse).
    """
    if norm not in ('l1', 'l2'):
        raise ValueError("'%s' is not a supported norm. Currently supported norms are 'l1' and 'l2'." % norm)
    if sparse.issparse(vec):
        vec = vec.tocsr()
        if norm == 'l1':
            veclen = np.sum(np.abs(vec.data))
        if norm == 'l2':
            veclen = np.sqrt(np.sum(vec.data ** 2))
        if veclen > 0.0:
            return vec / veclen
        else:
            return vec

    if isinstance(vec, np.ndarray):
        vec = np.asarray(vec, dtype=float)
        if norm == 'l1':
            veclen = np.sum(np.abs(vec))
        if norm == 'l2':
            veclen = np.linalg.norm(vec)
        if veclen > 0.0:
            return vec / veclen
        else:
            return vec

    try:
        first = next(iter(vec))     # is there at least one element?
    except:
        return vec

    if isinstance(first, (tuple, list)) and len(first) == 2: # gensim sparse format
        if norm == 'l1':
            length = float(sum(abs(val) for _, val in vec))
        if norm == 'l2':
            length = 1.0 * math.sqrt(sum(val ** 2 for _, val in vec))
        assert length > 0.0, "sparse documents must not contain any explicit zero entries"
        return ret_normalized_vec(vec, length)
    else:
        raise ValueError("unknown input type")

def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.
    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.
    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.
    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg. : ["coef_", "estimator_", ...], "coef_"
    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.
    """
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})

class NotFittedError(ValueError, AttributeError):
    pass

def calc_accuracy(sys_out, ground):
    assert len(sys_out) == len(ground)
    n = len(sys_out)
    return sum([sys_out[i] == ground[i] for i in range(n)]) / float(n)

def load_pickle(path_to_file):
    try:
        data = pickle.load(open(path_to_file, 'r'))
    except Exception as e:
        raise e

    return data

def get_emb(emb_file, labels):
    sys.path.append("../autoextend")
    from embedding import PreTrainEmbedding
    pt = PreTrainEmbedding(emb_file, None)
    label_dict = {}
    for each in labels:
        core = each.split(':')[-1].lower()
        emb = pt.get_embedding(core)
        if emb == None:
            try:
                emb = np.average(np.r_[[pt.get_embedding(x) for x in core.split('_') if x in pt.model]], axis=0)
            except Exception as e:
                raise e
        if type(emb) == type(np.zeros(1)):
            label_dict[each] = emb

def get_emb2(emb_file, vocab):
    sys.path.append("../autoextend")
    from embedding import PreTrainEmbedding
    pt = PreTrainEmbedding(emb_file, None)
    vocab_dict = {}
    for each in vocab:
        core = each.lower()
        emb = pt.get_embedding(core)
        if emb == None:
            try:
                emb = np.average(np.r_[[pt.get_embedding(x) for x in core.split('_') if x in pt.model]], axis=0)
            except Exception as e:
                raise e
        if type(emb) == type(np.zeros(1)):
            vocab_dict[each] = emb

    return vocab_dict


