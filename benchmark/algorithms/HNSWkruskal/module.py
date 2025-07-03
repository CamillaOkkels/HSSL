from benchmark.algorithms.base.module import BaseClustering
from HSSL import *
import numpy as np
import json
import graphidxbaselines as gib
from benchmark.algorithms.default_hnsw_params import DEFAULT_PARAMS

# Only for the server!!!
try: gib.limit_threads(50)
except: pass

class HNSWkruskal(BaseClustering):
    def __init__(self, minPts, symmetric_expand, max_build_heap_size, lowest_max_degree):
        self.params = {**DEFAULT_PARAMS} # Copy default params
        self.params["max_build_heap_size"] = max_build_heap_size
        self.params["higher_max_degree"] = lowest_max_degree//2
        self.params["lowest_max_degree"] = lowest_max_degree
        self.symmetric_expand = symmetric_expand
        self.minPts = minPts

    def cluster(self, X: np.array):
        self.dendrogram = gib.graph_based_dendrogram(
            X,
            min_pts = self.minPts,
            expand=True,
            symmetric_expand = self.symmetric_expand,
            **self.params,
        )

    def retrieve_dendrogram(self):
        return self.dendrogram[0]
    def retrieve_milestones(self):
        return self.dendrogram[2]
    
    def __str__(self):
        return json.dumps(dict(
            minPts = self.minPts,
            symmetric_expand = self.symmetric_expand,
            params = self.params,
        ))

    def __repr__(self):
        return "{:}_{:}_{:}_{:}".format(
            self.minPts,
            self.symmetric_expand,
            self.params['max_build_heap_size'],
            self.params['lowest_max_degree'],
        )