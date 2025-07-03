import json
import numpy as np
import math
from benchmark.algorithms.base.module import BaseClustering
from lshlink import LSHLink

class LSHSingleLinkage(BaseClustering):
    def __init__(self, A, k, l):
        # def __init__(self, ef, max_build_heap_size, lowest_max_degree):
        self.A = A
        self.l = l
        self.k = k
        pass

    def cluster(self, X: np.array):
        X_scaled = (X - X.min()) / (X.max() - X.min())
        _, self.dendrogram = LSHLink(X_scaled,
                                     self.A, self.l, self.k, C = int(np.ceil(np.max(X_scaled) * 1.5)),
                                     dendrogram=True)

    def retrieve_dendrogram(self):
        return self.dendrogram
    
    def __str__(self):
        return json.dumps(dict(
            A = self.A,
            l = self.l,
            k = self.k
        ))

    def __repr__(self):
        return "{:}_{:}_{:}".format(
            self.A,
            self.l,
            self.k
        )