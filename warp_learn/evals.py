'''
Created on Dec, 2016

@author: hugo

'''

def is_hit(true, pred):
    if isinstance(true, str):
        true = [true]
    elif isinstance(true, (list, tuple)):
        pass
    else:
        raise TypeError('Unknown argument true:%s' % true)
    for each_true in true:
        each_true = each_true.split(':')
        for each in each_true:
            if each.lower() in pred:
                return True
    return False

def hit_rank(true, pred):
    if isinstance(true, str):
        true = [true]
    elif isinstance(true, (list, tuple)):
        pass
    else:
        raise TypeError('Unknown argument true:%s' % true)

    true_labels = list(set([y.lower() for x in true for y in x.split(':')]))
    rank = float('inf')
    for each in true_labels:
        if each in pred:
            tmp = pred.index(each)
            if rank > tmp:
                rank = tmp
    return rank if rank != float('inf') else None


def match_at_K(truth, results, K):
    """
    Match@K: The relative number of clusters for which at least one of the top-K labels is correct.

    @params
    truth : dict, key is clus name, value is a list of true labels (we can add labels which we think are correct)
    results : dict, key is clus name, value if a list of predicted labels
    K : the K in the definition.
    """
    hit_count = 0.
    for k, v in results.items():
        if is_hit(truth[k], v[:K]):
            hit_count += 1
    return hit_count / len(results)

def mrr_at_K(truth, results, K):
    """
    Mean Reciprocal Rank (MRR@K): Given an ordered list of K proposed labels for a cluster, the reciprocal rank
    is the inverse of the rank of the first correct label, or zero if no label in the list is correct. The mean
    reciprocal rank at K (MRR@K) is the average of the reciprocal ranks of all clusters.

    @params
    truth : dict, key is clus name, value is a list of true labels (we can add labels which we think are correct)
    results : dict, key is clus name, value if a list of predicted labels
    K : the K in the definition.
    """
    score = 0.
    for k, v in results.items():
        rank = hit_rank(truth[k], v[:K])
        if rank != None:
            score += 1 / (1. + rank)

    return score / len(results)
