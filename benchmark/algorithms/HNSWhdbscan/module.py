from benchmark.algorithms.base.module import BaseClustering
from HSSL import *
import json

class HNSWhdbscan(BaseClustering):
    def __init__(self, minPts, higher_max_degree, lowest_max_degree):
        self.minPts = minPts
        self.higher_max_degree = higher_max_degree
        self.lowest_max_degree = lowest_max_degree
        pass

    def cluster(self, X: np.array):  
        self.dendrogram = gib.graph_based_dendrogram(
            X, 
            min_pts = self.minPts, 
            symmetric_expand=True, 
            higher_max_degree=self.higher_max_degree, 
            lowest_max_degree=self.lowest_max_degree
        )

    def retrieve_dendrogram(self):
        return self.dendrogram
    
    def __str__(self):
        return json.dumps(dict(
            minPts = self.minPts,
            higher_max_degree=self.higher_max_degree,
            lowest_max_degree=self.lowest_max_degree
        ))

    def __repr__(self):
        return f"{self.minPts}_{self.higher_max_degree}_{self.lowest_max_degree}"