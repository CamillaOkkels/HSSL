from benchmark.algorithms.base.module import BaseClustering
from sklearn.cluster import AgglomerativeClustering

import numpy as np

class SKLearnSingleLinkage(BaseClustering):
    def __init__(self):
        pass

    def cluster(self, X: np.array):  
        self.clustering = AgglomerativeClustering(n_clusters=1, linkage='single').fit(X)

    def retrieve_dendrogram(self):
        return self.clustering.children_
    
    def __str__(self):
        return f"SKLearnAgglomerativeClustering()"

    def __repr__(self):
        return f"run"