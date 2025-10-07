from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from benchmark.results import *
from scipy.stats import spearmanr, pearsonr
from HSSL import *
import pandas as pd
import seaborn as sns
import time
import itertools

import os, h5py
from typing import Optional

def count_files(dataset: Optional[str] = None, prefix: str = ".") -> int:
    count = 0
    for _, _, files in os.walk(os.path.join(prefix, build_result_filepath(dataset))):
        count += sum(1 for f in files if f.endswith(".hdf5"))
    return count

dataset = "aloi733-111k"
algo = "HNSWhssl"
print(count_files(dataset + "/" + algo + "/run=1"))
print(count_files(dataset + "/" + algo + "/run=2"))
print(count_files(dataset + "/" + algo + "/run=3"))
print(count_files(dataset + "/" + algo + "/run=4"))
print(count_files(dataset + "/" + algo + "/run=5"))
print(count_files(dataset + "/" + algo + "/run=6"))
print(count_files(dataset + "/" + algo + "/run=7"))

def Ground_Truth_coph_dist(dataset, pearson):
    if(pearson):
        with h5py.File("data/" + dataset + ".hdf5", "r") as f:
            d = f['data']
            d_array = d[:]
            dist_mat = None # dist_mat = pdist(d_array, metric='euclidean')

    for f in load_all_results(dataset, ""):
        try:
            if f.attrs['algo'] != 'scipy': continue
            print("Computing ground-truth dendrogram data...")
            gt_dendro = f["dendrogram"][:]
            gt_dendro = [[int(l), int(r), float(d), int(s)] for l, r, d, s in gt_dendro]
            start = time.time()
            gt_coph_dists = cophenet(gt_dendro)
            # if(pearson):    
                # coph_coeff_true, _ = cophenet(gt_dendro, dist_mat)
                # print(f"ground-truth cophenetic coefficient: {coph_coeff_true}")
            end = time.time()
            t = end - start
            print(f"gt_coph_dist took: {t}s")
        finally:
            f.close()
    indices = np.random.choice(gt_coph_dists.shape[0], size=10_000_000, replace=False)
    if(pearson):
        return gt_coph_dists, indices, dist_mat
    return gt_coph_dists, indices

def compute_quality(dataset, combinations, data, gt_coph_dists, measure, algo, indices, dist_mat=None):
    temp = 0
    error_count = 0

    with tqdm(total=count_files(dataset + "/" + algo), desc=f"Computing qualities ({measure}, {dataset}, {algo})...") as pbar:
        for f in load_all_results(dataset, ""):
            if f.attrs['algo'] != algo: continue
            params = json.loads(f.attrs["params"])
            isHSSL = True if algo == "HNSWhssl" else False
            
            if isHSSL:
                if (params['ef'], 
                params['params']['max_build_heap_size'], 
                params['params']['lowest_max_degree']) not in combinations: continue
            
            if not isHSSL:
                if (params['symmetric_expand'], 
                params['params']['max_build_heap_size'], 
                params['params']['lowest_max_degree']) not in combinations: continue
        
            try:
                dendro = [[int(l), int(r), float(d), int(s)] for l, r, d, s in f["dendrogram"][:]]

                try:
                    start = time.time()
                    coph_dists = cophenet(dendro)
                    end = time.time()
                    t = end - start
                    print(f"coph_dist took: {t}s")
                except:
                    print("ERROR in cophenet")
                    error_count += 1
                    continue

                sampled_gt_coph_dists = gt_coph_dists[indices]
                sampled_coph_dists = coph_dists[indices] if isHSSL else np.sqrt(coph_dists[indices])

                temp += 1
                if temp <= 10:
                    if isHSSL:
                        plt.figure(figsize=(6, 4))
                        sns.scatterplot(x=sampled_gt_coph_dists[0:10000], y=sampled_coph_dists[0:10000], alpha=0.3)
                        plt.title(f"Cophenetic distances: ef={params['ef']}, mbhs={params['params']['max_build_heap_size']}, lmd={params['params']['lowest_max_degree']}")
                        plt.xlabel("Ground Truth Cophenetic Distances")
                        plt.ylabel("Sampled Cophenetic Distances")
                        plt.grid(True)
                        plt.tight_layout()
                        plt.show()
                    if not isHSSL:
                        plt.figure(figsize=(6, 4))
                        sns.scatterplot(x=sampled_gt_coph_dists[0:10000], y=sampled_coph_dists[0:10000], alpha=0.3)
                        plt.title(f"Cophenetic distances: symmetric_expand={params['symmetric_expand']}, mbhs={params['params']['max_build_heap_size']}, lmd={params['params']['lowest_max_degree']}")
                        plt.xlabel("Ground Truth Cophenetic Distances")
                        plt.ylabel("Sampled Cophenetic Distances")
                        plt.grid(True)
                        plt.tight_layout()
                        plt.show()



                if(measure == "pearson"):
                    compute_quality_pearson(data, dist_mat, f, dendro, sampled_gt_coph_dists, sampled_coph_dists)
                elif(measure == "norm"):
                    compute_quality_norm(data, f, sampled_gt_coph_dists, sampled_coph_dists)
                elif(measure == "mean_ratio"):
                    compute_quality_mean_ratio(data, f, sampled_gt_coph_dists, sampled_coph_dists)
                else:
                    print("No measure found.")

            finally:
                f.close()
            pbar.update(1)
            pbar.refresh()

    return error_count


def compute_quality_pearson(data, dist_mat, f, dendro, sampled_gt_coph_dists, sampled_coph_dists):
    start = time.time()
    pear_corr, p_val = pearsonr(sampled_gt_coph_dists, sampled_coph_dists)
    # coph_coeff, _ = cophenet(dendro, dist_mat)
    print(f"pearson correlation: {pear_corr}")
    # print(f"cophenetic coefficient: {coph_coeff}")
    end = time.time()
    t = end - start
    print(f"Pearson took: {t}s")

    data.append({
        "algo": f.attrs['algo'],
        "time": f.attrs['time'],
        "n": len(f["dendrogram"][:]) + 1,
        "params": f.attrs["params"],
        "run": f.attrs["run"],
        "Pearson Correlation": pear_corr,
        "pval": p_val
    })

def compute_quality_norm(data, f, sampled_gt_coph_dists, sampled_coph_dists):
    start = time.time()
    max_diff = max(sampled_gt_coph_dists) - min(sampled_gt_coph_dists)
    norm = np.linalg.norm(sampled_gt_coph_dists - sampled_coph_dists) / ( np.sqrt(len(sampled_gt_coph_dists)) * max_diff )
    print(norm)
    end = time.time()
    t = end - start
    print(f"norm took: {t}s")

    data.append({
            "algo": f.attrs['algo'],
            "time": f.attrs['time'],
            "n": len(f["dendrogram"][:]) + 1,
            "params": f.attrs["params"],
            "run": f.attrs["run"],
            "norm": 1 - norm, 
            })

def compute_quality_mean_ratio(data, f, sampled_gt_coph_dists, sampled_coph_dists):
    mask1 = sampled_gt_coph_dists != 0
    sampled_gt_coph_dists = sampled_gt_coph_dists[mask1]
    sampled_coph_dists = sampled_coph_dists[mask1]
    mask2 = sampled_coph_dists != 0
    sampled_gt_coph_dists = sampled_gt_coph_dists[mask2]
    sampled_coph_dists = sampled_coph_dists[mask2]
            
    start = time.time()
    ratio = [x / y for x, y in zip(sampled_coph_dists, sampled_gt_coph_dists)]
    mean_ratio = np.exp(np.mean(np.abs(np.log(ratio))))
    print(mean_ratio)
    end = time.time()
    t = end - start
    print(f"mean_ratio took: {t}s")

    # np.savez_compressed(f.filename.split("/")[4].split(".")[0] + "_ratio_data.npz", ratio=ratio, mean_ratio=mean_ratio)

    # plt.figure(figsize=(8, 5))
    # plt.hist(ratio, bins=1000, edgecolor="black", alpha=0.7)
    # plt.axvline(mean_ratio, color="red", linestyle="--", linewidth=2, label=f"Mean Ratio = {mean_ratio:.3f}")
    # plt.xlabel("Ratio values")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Ratio Values")
    # plt.legend()
    # plt.show()

    data.append({
            "algo": f.attrs['algo'],
            "time": f.attrs['time'],
            "n": len(f["dendrogram"][:]) + 1,
            "params": f.attrs["params"],
            "run": f.attrs["run"],
            "mean_ratio": mean_ratio,
            })
    

ef = [5, 11, 22, 47, 100]
mbhs = [25, 42, 71, 119, 200]
lmd = [14, 26, 51, 100]

se = [False]

combinations_hssl = list(itertools.product(ef, mbhs, lmd))
combinations_nothssl = list(itertools.product(se, mbhs, lmd))

algorithms = ["HNSWhssl", "HNSWkruskal", "HNSWmst"]


if 1: ### --- run pearson --- ####
    measure = "pearson"
    run_pearson = True
    
    if 1: ### ---- run ALOI ---- ###
        
        dataset = 'aloi733-111k'

        gt_coph_dists, indices, dist_mat = Ground_Truth_coph_dist(dataset, run_pearson)

        if 1: ### ---- hssl ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_hssl, data, gt_coph_dists, measure, algorithms[0], indices, dist_mat)

            print(error_count)
            df_ALOIhssl = pd.DataFrame(data=data)
            df_ALOIhssl.head(5)

            df_ALOIhssl_split = df_ALOIhssl
            df_ALOIhssl_split["params_dict"] = df_ALOIhssl_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_ALOIhssl_split["params_dict"].apply(lambda d: d.get("ef")).rename("ef").to_frame()
            df_params = df_ALOIhssl_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_ALOIhssl_split = pd.concat([df_ALOIhssl_split, df_ef[['ef']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_ALOIhssl_split.to_csv('evaluation/ALOIhssl_pearson.csv', index=False)

        if 1: ### ---- kruskal ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_nothssl, data, gt_coph_dists, measure, algorithms[1], indices, dist_mat)

            print(error_count)
            df_ALOIkruskal = pd.DataFrame(data=data)
            df_ALOIkruskal.head(5)

            df_ALOIkruskal_split = df_ALOIkruskal
            df_ALOIkruskal_split["params_dict"] = df_ALOIkruskal_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_ALOIkruskal_split["params_dict"].apply(lambda d: d.get("symmetric_expand")).rename("symmetric_expand").to_frame()
            df_params = df_ALOIkruskal_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_ALOIkruskal_split = pd.concat([df_ALOIkruskal_split, df_ef[['symmetric_expand']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_ALOIkruskal_split.to_csv('evaluation/ALOIkruskal_pearson.csv', index=False)

        if 1: ### ---- mst ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_nothssl, data, gt_coph_dists, measure, algorithms[2], indices, dist_mat)

            print(error_count)
            df_ALOImst = pd.DataFrame(data=data)
            df_ALOImst.head(5)

            df_ALOImst_split = df_ALOImst
            df_ALOImst_split["params_dict"] = df_ALOImst_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_ALOImst_split["params_dict"].apply(lambda d: d.get("symmetric_expand")).rename("symmetric_expand").to_frame()
            df_params = df_ALOImst_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_ALOImst_split = pd.concat([df_ALOImst_split, df_ef[['symmetric_expand']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_ALOImst_split.to_csv('evaluation/ALOImst_pearson.csv', index=False)
    

    if 1: ### ---- run MNIST ---- ###
        
        dataset = 'mnist-70k'

        gt_coph_dists, indices, dist_mat = Ground_Truth_coph_dist(dataset, run_pearson)

        if 1: ### ---- hssl ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_hssl, data, gt_coph_dists, measure, algorithms[0], indices, dist_mat)

            print(error_count)
            df_MNISThssl = pd.DataFrame(data=data)
            df_MNISThssl.head(5)

            df_MNISThssl_split = df_MNISThssl
            df_MNISThssl_split["params_dict"] = df_MNISThssl_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_MNISThssl_split["params_dict"].apply(lambda d: d.get("ef")).rename("ef").to_frame()
            df_params = df_MNISThssl_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_MNISThssl_split = pd.concat([df_MNISThssl_split, df_ef[['ef']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_MNISThssl_split.to_csv('evaluation/MNISThssl_pearson.csv', index=False)

        if 1: ### ---- kruskal ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_nothssl, data, gt_coph_dists, measure, algorithms[1], indices, dist_mat)

            print(error_count)
            df_MNISTkruskal = pd.DataFrame(data=data)
            df_MNISTkruskal.head(5)

            df_MNISTkruskal_split = df_MNISTkruskal
            df_MNISTkruskal_split["params_dict"] = df_MNISTkruskal_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_MNISTkruskal_split["params_dict"].apply(lambda d: d.get("symmetric_expand")).rename("symmetric_expand").to_frame()
            df_params = df_MNISTkruskal_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_MNISTkruskal_split = pd.concat([df_MNISTkruskal_split, df_ef[['symmetric_expand']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_MNISTkruskal_split.to_csv('evaluation/MNISTkruskal_pearson.csv', index=False)

        if 1: ### ---- mst ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_nothssl, data, gt_coph_dists, measure, algorithms[2], indices, dist_mat)

            print(error_count)
            df_MNISTmst = pd.DataFrame(data=data)
            df_MNISTmst.head(5)

            df_MNISTmst_split = df_MNISTmst
            df_MNISTmst_split["params_dict"] = df_MNISTmst_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_MNISTmst_split["params_dict"].apply(lambda d: d.get("symmetric_expand")).rename("symmetric_expand").to_frame()
            df_params = df_MNISTmst_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_MNISTmst_split = pd.concat([df_MNISTmst_split, df_ef[['symmetric_expand']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_MNISTmst_split.to_csv('evaluation/MNISTmst_pearson.csv', index=False)


if 0: ### --- run norm --- ####
    measure = "norm"
    run_pearson = False
    
    if 1: ### ---- run ALOI ---- ###
        
        dataset = 'aloi733-111k'

        gt_coph_dists, indices = Ground_Truth_coph_dist(dataset, run_pearson)

        if 0: ### ---- hssl ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_hssl, data, gt_coph_dists, measure, algorithms[0], indices)

            print(error_count)
            df_ALOIhssl = pd.DataFrame(data=data)
            df_ALOIhssl.head(5)

            df_ALOIhssl_split = df_ALOIhssl
            df_ALOIhssl_split["params_dict"] = df_ALOIhssl_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_ALOIhssl_split["params_dict"].apply(lambda d: d.get("ef")).rename("ef").to_frame()
            df_params = df_ALOIhssl_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_ALOIhssl_split = pd.concat([df_ALOIhssl_split, df_ef[['ef']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_ALOIhssl_split.to_csv('evaluation/ALOIhssl_norm.csv', index=False)

        if 0: ### ---- kruskal ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_nothssl, data, gt_coph_dists, measure, algorithms[1], indices)

            print(error_count)
            df_ALOIkruskal = pd.DataFrame(data=data)
            df_ALOIkruskal.head(5)

            df_ALOIkruskal_split = df_ALOIkruskal
            df_ALOIkruskal_split["params_dict"] = df_ALOIkruskal_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_ALOIkruskal_split["params_dict"].apply(lambda d: d.get("symmetric_expand")).rename("symmetric_expand").to_frame()
            df_params = df_ALOIkruskal_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_ALOIkruskal_split = pd.concat([df_ALOIkruskal_split, df_ef[['symmetric_expand']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_ALOIkruskal_split.to_csv('evaluation/ALOIkruskal_norm.csv', index=False)

        if 1: ### ---- mst ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_nothssl, data, gt_coph_dists, measure, algorithms[2], indices)

            print(error_count)
            df_ALOImst = pd.DataFrame(data=data)
            df_ALOImst.head(5)

            df_ALOImst_split = df_ALOImst
            df_ALOImst_split["params_dict"] = df_ALOImst_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_ALOImst_split["params_dict"].apply(lambda d: d.get("symmetric_expand")).rename("symmetric_expand").to_frame()
            df_params = df_ALOImst_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_ALOImst_split = pd.concat([df_ALOImst_split, df_ef[['symmetric_expand']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_ALOImst_split.to_csv('evaluation/ALOImst_norm.csv', index=False)


    if 1: ### ---- run MNIST ---- ###
        
        dataset = 'mnist-70k'

        gt_coph_dists, indices = Ground_Truth_coph_dist(dataset, run_pearson)

        if 1: ### ---- hssl ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_hssl, data, gt_coph_dists, measure, algorithms[0], indices)

            print(error_count)
            df_MNISThssl = pd.DataFrame(data=data)
            df_MNISThssl.head(5)

            df_MNISThssl_split = df_MNISThssl
            df_MNISThssl_split["params_dict"] = df_MNISThssl_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_MNISThssl_split["params_dict"].apply(lambda d: d.get("ef")).rename("ef").to_frame()
            df_params = df_MNISThssl_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_MNISThssl_split = pd.concat([df_MNISThssl_split, df_ef[['ef']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_MNISThssl_split.to_csv('evaluation/MNISThssl_norm.csv', index=False)

        if 1: ### ---- kruskal ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_nothssl, data, gt_coph_dists, measure, algorithms[1], indices)

            print(error_count)
            df_MNISTkruskal = pd.DataFrame(data=data)
            df_MNISTkruskal.head(5)
            
            df_MNISTkruskal_split = df_MNISTkruskal
            df_MNISTkruskal_split["params_dict"] = df_MNISTkruskal_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_MNISTkruskal_split["params_dict"].apply(lambda d: d.get("symmetric_expand")).rename("symmetric_expand").to_frame()
            df_params = df_MNISTkruskal_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_MNISTkruskal_split = pd.concat([df_MNISTkruskal_split, df_ef[['symmetric_expand']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_MNISTkruskal_split.to_csv('evaluation/MNISTkruskal_norm.csv', index=False)

        if 1: ### ---- mst ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_nothssl, data, gt_coph_dists, measure, algorithms[2], indices)

            print(error_count)
            df_MNISTmst = pd.DataFrame(data=data)
            df_MNISTmst.head(5)

            df_MNISTmst_split = df_MNISTmst
            df_MNISTmst_split["params_dict"] = df_MNISTmst_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_MNISTmst_split["params_dict"].apply(lambda d: d.get("symmetric_expand")).rename("symmetric_expand").to_frame()
            df_params = df_MNISTmst_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_MNISTmst_split = pd.concat([df_MNISTmst_split, df_ef[['symmetric_expand']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_MNISTmst_split.to_csv('evaluation/MNISTmst_norm.csv', index=False)


if 0: ### --- run mean_ratio --- ####
    measure = "mean_ratio"
    run_pearson = False
    
    if 1: ### ---- run ALOI ---- ###
        
        dataset = 'aloi733-111k'

        gt_coph_dists, indices = Ground_Truth_coph_dist(dataset, run_pearson)

        if 1: ### ---- hssl ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_hssl, data, gt_coph_dists, measure, algorithms[0], indices)

            print(error_count)
            df_ALOIhssl = pd.DataFrame(data=data)
            df_ALOIhssl.head(5)
            
            df_ALOIhssl_split = df_ALOIhssl
            df_ALOIhssl_split["params_dict"] = df_ALOIhssl_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_ALOIhssl_split["params_dict"].apply(lambda d: d.get("ef")).rename("ef").to_frame()
            df_params = df_ALOIhssl_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_ALOIhssl_split = pd.concat([df_ALOIhssl_split, df_ef[['ef']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_ALOIhssl_split.to_csv('evaluation/ALOIhssl_mean_ratio.csv', index=False)

        if 1: ### ---- kruskal ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_nothssl, data, gt_coph_dists, measure, algorithms[1], indices)

            print(error_count)
            df_ALOIkruskal = pd.DataFrame(data=data)
            df_ALOIkruskal.head(5)
            
            df_ALOIkruskal_split = df_ALOIkruskal
            df_ALOIkruskal_split["params_dict"] = df_ALOIkruskal_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_ALOIkruskal_split["params_dict"].apply(lambda d: d.get("symmetric_expand")).rename("symmetric_expand").to_frame()
            df_params = df_ALOIkruskal_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_ALOIkruskal_split = pd.concat([df_ALOIkruskal_split, df_ef[['symmetric_expand']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_ALOIkruskal_split.to_csv('evaluation/ALOIkruskal_mean_ratio.csv', index=False)

        if 1: ### ---- mst ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_nothssl, data, gt_coph_dists, measure, algorithms[2], indices)

            print(error_count)
            df_ALOImst = pd.DataFrame(data=data)
            df_ALOImst.head(5)
            
            df_ALOImst_split = df_ALOImst
            df_ALOImst_split["params_dict"] = df_ALOImst_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_ALOImst_split["params_dict"].apply(lambda d: d.get("symmetric_expand")).rename("symmetric_expand").to_frame()
            df_params = df_ALOImst_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_ALOImst_split = pd.concat([df_ALOImst_split, df_ef[['symmetric_expand']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_ALOImst_split.to_csv('evaluation/ALOImst_mean_ratio.csv', index=False)
    

    if 1: ### ---- run MNIST ---- ###
        
        dataset = 'mnist-70k'

        gt_coph_dists, indices = Ground_Truth_coph_dist(dataset, run_pearson)

        if 1: ### ---- hssl ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_hssl, data, gt_coph_dists, measure, algorithms[0], indices)

            print(error_count)
            df_MNISThssl = pd.DataFrame(data=data)
            df_MNISThssl.head(5)
            
            df_MNISThssl_split = df_MNISThssl
            df_MNISThssl_split["params_dict"] = df_MNISThssl_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_MNISThssl_split["params_dict"].apply(lambda d: d.get("ef")).rename("ef").to_frame()
            df_params = df_MNISThssl_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_MNISThssl_split = pd.concat([df_MNISThssl_split, df_ef[['ef']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_MNISThssl_split.to_csv('evaluation/MNISThssl_mean_ratio.csv', index=False)

        if 1: ### ---- kruskal ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_nothssl, data, gt_coph_dists, measure, algorithms[1], indices)

            print(error_count)
            df_MNISTkruskal = pd.DataFrame(data=data)
            df_MNISTkruskal.head(5)
            
            df_MNISTkruskal_split = df_MNISTkruskal
            df_MNISTkruskal_split["params_dict"] = df_MNISTkruskal_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_MNISTkruskal_split["params_dict"].apply(lambda d: d.get("symmetric_expand")).rename("symmetric_expand").to_frame()
            df_params = df_MNISTkruskal_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_MNISTkruskal_split = pd.concat([df_MNISTkruskal_split, df_ef[['symmetric_expand']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_MNISTkruskal_split.to_csv('evaluation/MNISTkruskal_mean_ratio.csv', index=False)

        if 1: ### ---- mst ---- ###
            data = []
            error_count = compute_quality(dataset, combinations_nothssl, data, gt_coph_dists, measure, algorithms[2], indices)

            print(error_count)
            df_MNISTmst = pd.DataFrame(data=data)
            df_MNISTmst.head(5)

            df_MNISTmst_split = df_MNISTmst
            df_MNISTmst_split["params_dict"] = df_MNISTmst_split["params"].apply(lambda x: json.loads(x))
            df_ef = df_MNISTmst_split["params_dict"].apply(lambda d: d.get("symmetric_expand")).rename("symmetric_expand").to_frame()
            df_params = df_MNISTmst_split["params_dict"].apply(lambda d: d.get("params", {})).apply(pd.Series)
            df_MNISTmst_split = pd.concat([df_MNISTmst_split, df_ef[['symmetric_expand']], df_params[["lowest_max_degree", "max_build_heap_size"]]], axis=1)

            df_MNISTmst_split.to_csv('evaluation/MNISTmst_mean_ratio.csv', index=False)


