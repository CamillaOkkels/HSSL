from benchmark.algorithms.base.module import BaseClustering
import subprocess
import pandas as pd
import numpy as np
import json
import time

# java -jar elki-bundle-0.8.1-SNAPSHOT.jar cli -time -algorithm clustering.hierarchical.extraction.SimplifiedHierarchyExtraction -algorithm HSSL -dbc.in /mnt/large_storage/cabi/hnsw_searchers/HSSL/Aloi733.csv.gz -resulthandler DiscardResultHandler -evaluator NoAutomaticEvaluation

class VPTreeJavaSingleLinkage(BaseClustering):
    def __init__(self):
        pass

    def cluster(self, X: np.array):
        start = time.time()
        df = pd.DataFrame(X)
        df.to_csv("/tmp/data.csv")
        end = time.time()
        self.overhead = end - start
        print(f"Converting took {end - start}s.")
        cmd = "java -jar elki-bundle-0.8.1-SNAPSHOT.jar cli -time -algorithm clustering.hierarchical.extraction.SimplifiedHierarchyExtraction -algorithm HSSL -dbc.in /tmp/data.csv -resulthandler DiscardResultHandler -evaluator NoAutomaticEvaluation".split()
        self.dendrogram = subprocess.run(cmd, capture_output=True, text=True).stdout
        print(self.dendrogram)

    def retrieve_dendrogram(self):
        return self.dendrogram
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
        return json.dumps({})

    def __repr__(self):
        return "VPTreeJava"

    def get_overhead_time(self):
        return self.overhead

