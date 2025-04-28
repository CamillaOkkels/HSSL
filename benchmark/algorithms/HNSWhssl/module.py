from benchmark.algorithms.base.module import BaseClustering
from HSSL import *

class HNSWSingleLinkage(BaseClustering):
    def __init__(self, ef, max_build_heap_size, max_degree):
        self.ef = ef
        self.max_build_heap_size = max_build_heap_size
        self.max_degree = max_degree
        pass

    def cluster(self, X: np.array):  
        self.dendrogram = HNSW_HSSL(X, ef=self.ef, 
                                    max_build_heap_size=self.max_build_heap_size, 
                                    higher_max_degree=self.max_degree)

    def retrieve_dendrogram(self):
        return self.dendrogram
    
    def __str__(self):
        return f"HNSWHierarcihcalSingleLinkage()"

    def __repr__(self):
        return f"{self.ef}_{self.max_build_heap_size}_{self.max_degree}"