'''
Created on Nov, 2016

@author: hugo

'''

import cPickle as pickle

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
    import sys
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

    return label_dict
