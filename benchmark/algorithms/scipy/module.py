from benchmark.algorithms.base.module import BaseClustering
import scipy

import numpy as np

class SciPySingleLinkage(BaseClustering):
    def __init__(self):
        pass

    def cluster(self, X: np.array):  
        self.clustering = scipy.cluster.hierarchy.single(X)

    def retrieve_dendrogram(self):
        return self.clustering
    def retrieve_milestones(self):
        milestones = {
        0.25: None,
        0.5: None,
        0.75: None,
        0.8: None,
        0.9: None,
        0.95: None,
        0.99: None,
    }
        return milestones

    def __str__(self):
        return f"SciPyHierarchicalClustering()"

    def __repr__(self):
        return f"run"