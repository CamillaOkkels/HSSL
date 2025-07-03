from benchmark.algorithms.base.module import BaseClustering
from HSSL import *
import json

class VPTreeSingleLinkage(BaseClustering):
    def __init__(self, n_trees, clean_fraction):
        self.n_trees = n_trees
        self.clean_fraction = clean_fraction

    def cluster(self, X: np.array):  
        self.dendrogram = HSSL_Turbo(
            X,
            n_trees=self.n_trees,
            cuda=False,
            clean_fraction=self.clean_fraction,
        )

    def retrieve_dendrogram(self):
        return self.dendrogram
    
    def __str__(self):
        return json.dumps(dict(
            n_trees=self.n_trees,
            clean_fraction=self.clean_fraction,
        ))

    def __repr__(self):
        return "{:}_{:}".format(
            self.n_trees,
            self.clean_fraction,
        )
