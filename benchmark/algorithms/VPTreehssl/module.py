from benchmark.algorithms.base.module import BaseClustering
from HSSL import *

class VPTreeSingleLinkage(BaseClustering):
    def __init__(self):
        pass

    def cluster(self, X: np.array):  
        self.dendrogram = HSSL_Turbo(X, n_trees=1, cuda=False, clean_fraction=2)

    def retrieve_dendrogram(self):
        return self.dendrogram
    
    def __str__(self):
        return f"VPTreeHierarchicalSingleLinkage()"

    def __repr__(self):
        return f"run"