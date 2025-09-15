from benchmark.algorithms.base.module import BaseClustering
from HSSL import *
import json
from benchmark.algorithms.default_hnsw_params import DEFAULT_PARAMS

# Only for the server!!!
try: gib.limit_threads(50)
except: pass

class HNSWSingleLinkage(BaseClustering):
    def __init__(self, ef, max_build_heap_size, lowest_max_degree):
        self.params = {**DEFAULT_PARAMS} # Copy default params
        self.params["max_build_heap_size"] = max_build_heap_size
        self.params["higher_max_degree"] = lowest_max_degree//2
        self.params["lowest_max_degree"] = lowest_max_degree
        self.ef = ef
        pass

    def cluster(self, X: np.array):  
        self.dendrogram, self.milestones = HNSW_HSSL(
            X, 
            ef = self.ef, 
            **self.params,
        )

    def retrieve_dendrogram(self):
        return self.dendrogram
    def retrieve_milestones(self):
        return self.milestones
    
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
