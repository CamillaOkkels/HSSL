from benchmark.algorithms.base.module import BaseClustering
from HSSL import *
import numpy as np
import json
import graphidxbaselines as gib
from benchmark.algorithms.default_hnsw_params import DEFAULT_PARAMS

class HNSWSingleLinkageRust(BaseClustering):
    def __init__(self, ef, max_build_heap_size, lowest_max_degree):
        self.params = {**DEFAULT_PARAMS} # Copy default params
        self.params["max_build_heap_size"] = max_build_heap_size
        self.params["higher_max_degree"] = lowest_max_degree//2
        self.params["lowest_max_degree"] = lowest_max_degree
        self.ef = ef
        pass

    def cluster(self, X: np.array):  
        self.dendrogram = gib.hnsw_hssl(
            X, 
            ef = self.ef, 
            **self.params,
        )

    def retrieve_dendrogram(self):
        return self.dendrogram[0]
    def retrieve_milestones(self):
        return self.dendrogram[1]
    
    def __str__(self):
        return json.dumps(dict(
            ef = self.ef,
            params = self.params,
        ))
        # return f"HNSW_HSSL(ef={self.ef}, ef_construct={self.max_build_heap_size}, M={self.max_degree})"

    def __repr__(self):
        return "{:}_{:}_{:}".format(
            self.ef,
            self.params['max_build_heap_size'],
            self.params['lowest_max_degree'],
        )
