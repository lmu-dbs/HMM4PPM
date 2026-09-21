import numpy as np
import itertools
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from concurrent.futures import ProcessPoolExecutor

import logging
logger = logging.getLogger(__name__)

class Evaluator:

    def __init__(self, pred: list[int], actual: list[int]):
        self.pred = pred
        self.actual = actual
        self.no_none_pred = [p if p is not None else -1 for p in self.pred]
    
    def get_nan_share(self) -> float:
        none_share = sum([pred is None for pred in self.pred])/len(self.pred)
        return none_share
    
    def _batch_seqs(self, seqs: list[list[int]], nbatches: int):
        nseqs = len(seqs)
        batchsize = math.ceil(nseqs/nbatches)
        for ndx in range(0, nseqs, batchsize):
            yield seqs[ndx:min(ndx + batchsize, nseqs)]

    def calc_accuracy_score(self) -> float:
        acc_score = accuracy_score(self.actual, self.no_none_pred)
        return acc_score

    def calc_balanced_accuracy_score(self) -> float:
        balanced_acc_score = balanced_accuracy_score(self.actual, self.no_none_pred)
        return balanced_acc_score

    def calc_ndls(self, horizon: int|None = None, ncores: int = 1) -> float:
        ndls_values = list()
        if ncores==1:
            for pred, actual in zip(self.pred, self.actual):
                try:
                    ndls_values.append(normalized_damerau_levenshtein_similarity(pred=pred, actual=actual, horizon=horizon))
                except TypeError:
                    ndls_values.append(0)
            ndls = sum(ndls_values)/len(self.pred)
        else:
            pred_batches = self._batch_seqs(self.pred, nbatches=ncores)
            actual_batches = self._batch_seqs(self.actual, nbatches=ncores)
            with ProcessPoolExecutor(max_workers=ncores) as executor:
                batch_ndls_values = executor.map(self._batch_calc_ndls,
                                                    pred_batches,
                                                    actual_batches,
                                                    [horizon]*ncores)
            ndls_values = list(itertools.chain(*batch_ndls_values))
            ndls = sum(ndls_values)/len(ndls_values)

        return ndls
    
    def _batch_calc_ndls(self, pred, actual, horizon):
        ndls_values = list()
        for pred, actual in zip(pred, actual):
            try:
                ndls_values.append(normalized_damerau_levenshtein_similarity(pred=pred, actual=actual, horizon=horizon))
            except TypeError:
                ndls_values.append(0)
        return ndls_values


def damerau_levenshtein_dist(pred: list, actual: list, horizon: int|None = None) -> float:
    if horizon:
        pred = [el for idx, el in enumerate(pred) if idx < horizon]
        actual = [el for idx, el in enumerate(actual) if idx < horizon]
    
    distance_mat = np.zeros((len(pred)+1, len(actual)+1))

    cp = {symbol:0 for symbol in set(pred + actual)}

    distance_mat[:,0] = range(0, len(pred)+1)
    distance_mat[0,:] = range(0, len(actual)+1)

    for i in range(0, len(pred)):
        
        cs = 0
        
        for j in range(0, len(actual)):

            if pred[i]==actual[j]:
                d = 0
            else:
                d = 1
            
            distance_mat[i+1, j+1] = min(distance_mat[i,j+1] + 1, distance_mat[i+1,j] + 1, distance_mat[i,j] + d)

            i_prime = cp[actual[j]]
            j_prime = cs

            if i_prime > 0 and j_prime > 0:
                distance_mat[i+1, j+1] = min(distance_mat[i+1,j+1], distance_mat[int(i_prime)-1, int(j_prime)-1]+(i+1-i_prime)+(j+1-j_prime)-1)
            
            if pred[i]==actual[j]:
                cs = j+1
        
        cp[pred[i]] = i+1

    dl_dist = distance_mat[len(pred), len(actual)]

    return dl_dist

def normalized_damerau_levenshtein_similarity(pred: list, actual: list, horizon: int|None = None) -> float:
    dl_dist = damerau_levenshtein_dist(pred=pred, actual=actual, horizon=horizon)
    normalized_dist = dl_dist/max(len(pred), len(actual))
    ndls = 1-normalized_dist

    return ndls