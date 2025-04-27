from benchmark.algorithms.base.module import BaseClustering
from HSSL import *

class HNSWSingleLinkage(BaseClustering):
    def __init__(self):
        pass

    def cluster(self, X: np.array):  
        self.dendrogram = HNSW_HSSL(X, ef=20)

    def retrieve_dendrogram(self):
        return self.dendrogram
    
    def __str__(self):
        return f"HNSWHierarcihcalSingleLinkage()"

    def __repr__(self):
        return f"run"