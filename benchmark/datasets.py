"""
Functionality to create datasets used in the evaluation.
"""
from glob import glob
import h5py
import os
import time
import numpy as np
import zipfile, sklearn


from typing import Dict, Tuple

from urllib.request import urlopen


def download(src: str, dst: str):
    """ download an URL """
    if os.path.exists(dst):
        #print("Already exists")
        return
    print('downloading %s -> %s...' % (src, dst))

    t0 = time.time()
    outf = open(dst, "wb")
    inf = urlopen(src)
    info = dict(inf.info())
    content_size = int(info.get('Content-Length', -1))
    bs = 1 << 20
    totsz = 0
    while True:
        block = inf.read(bs)
        elapsed = time.time() - t0
        print(
            "  [%.2f s] downloaded %.2f MiB / %.2f MiB at %.2f MiB/s   " % (
                elapsed,
                totsz / 2**20, content_size / 2**20 if content_size != -1 else -1,
                totsz / 2**20 / elapsed),
            flush=True, end="\r"
        )
        if not block:
            break
        outf.write(block)
        totsz += len(block)
    print()
    print("download finished in %.2f s, total size %d bytes" % (
        time.time() - t0, totsz
    ))

def get_dataset_fn(dataset_name: str) -> str:
    """
    Returns the full file path for a given dataset name in the data directory.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        str: The full file path of the dataset.
    """
    if not os.path.exists("data"):
        os.mkdir("data")
    return os.path.join("data", f"{dataset_name}.hdf5")


def get_dataset(dataset_name: str, path: str = ".") -> h5py.File:
    """
    Fetches a dataset by downloading it from a known URL or creating it locally
    if it's not already present. The dataset file is then opened for reading, 
    and the file handle and the dimension of the dataset are returned.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        Tuple[h5py.File, int]: A tuple containing the opened HDF5 file object and
            the dimension of the dataset.
    """
    hdf5_filename = get_dataset_fn(dataset_name)
    try:
        dataset_url = f"https://ann-benchmarks.com/{dataset_name}.hdf5"
        download(dataset_url, hdf5_filename)
    except:
        print(f"Cannot download {dataset_url}")
        if dataset_name in DATASETS:
            print("Creating dataset locally")
            DATASETS[dataset_name]['prepare']()#(hdf5_filename)

    hdf5_file = h5py.File(f"{path}/{hdf5_filename}", "r")

    return hdf5_file

def compute_groundtruth(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from benchmark.algorithms.sklearn.module import SKLearnSingleLinkage
    print("Computing groundtruth...")
    start = time.time()
    d = X.shape[1]
    clustering = SKLearnSingleLinkage()
    clustering.cluster(X)
    end = time.time()
    print(f"Computing groundtruth took {(end - start):.2f}s.")
    return clustering.retrieve_dendrogram()

def write_output(X: np.ndarray, name: str, compute_gt=True):
    f = h5py.File(get_dataset_fn(name), "w")
    f.create_dataset("data", data=X)
    if compute_gt:
        dendrogram = compute_groundtruth(X)
        f.create_dataset("dendrogram", data=dendrogram)
    f.close()

def mnist():
    from sklearn.datasets import fetch_openml
    if os.path.exists(get_dataset_fn("mnist")):
        return
    X,_ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    X = X.astype(np.float32)
    X /= np.sum(X, axis=1, keepdims=True)

    write_output(X, "mnist")


def pamap2(apply_pca=False):
    from sklearn.decomposition import PCA
    fn = "pamap2" if apply_pca else "pamap2-full"
    if os.path.exists(get_dataset_fn(fn)):
        return
    
    src = "http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
    download(src, "PAMAP2.zip")

    with zipfile.ZipFile("PAMAP2.zip") as zn:
        arr = []
        for i in range(1, 10):
            zfn = f"PAMAP2_Dataset/Protocol/subject10{i}.dat"
            zf = zn.open(zfn)
            for line in zf:
                line = line.decode()
                l = list(map(float, line.strip().split()))
                # remove timestamp
                arr.append(l[1:])
        X = np.nan_to_num(np.array(arr)) # many NaNs in data, replace them with 0.
        if apply_pca:
            X = PCA(n_components=4).fit_transform(X) # PCA of first four components
        write_output(X, fn)

    # PAMAP2_Dataset/Protocol/subject101.dat 

def household():
    # https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
    if os.path.exists(get_dataset_fn("household")):
        return

    src = 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'
    fn = "household.zip"
    download(src, "household.zip")

    with zipfile.ZipFile(fn) as z:
        zn = z.open('household_power_consumption.txt')
        zn.readline()
        cnt = []
        for line in zn:
            line = line.decode()
            if "?" not in line:
                cnt.append(list(map(float, line.strip().split(";")[2:])))
        X = np.array(cnt,dtype=np.float32)
        write_output(X, "household")

def aloi():
    if os.path.exists(get_dataset_fn("aloi")):
        return
    src = "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/1_ALOI.npz"
    download(src, "aloi.npz")
    X = np.load("aloi.npz")['X']
    write_output(X, "aloi")

def census():
    if os.path.exists(get_dataset_fn("census")):
        return
    src = "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/9_census.npz"
    download(src, "census.npz")
    X = np.load("census.npz")['X']
    write_output(X, "census")

def celeba():
    if os.path.exists(get_dataset_fn("celeba")):
        return
    src = "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/8_celeba.npz"
    download(src, "celeba.npz")
    X = np.load("celeba.npz")['X']
    write_output(X, "celeba")


def blobs(n, dim, centers):
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler

    name = f"blobs-{n // 1000}k-{dim}-{centers}"

    if os.path.exists(get_dataset_fn(name)):
        return

    X = make_blobs(n, dim, centers=centers, random_state=42)[0].astype(np.float32)

    write_output(np.array(X), name)   

DATASETS = {
    'mnist': {
        'prepare': mnist,
    }, 
    'pamap2': {
        'prepare': lambda: pamap2(True),
    },
    'pamap2-full': {
        'prepare': lambda: pamap2(False),
    },
    'household': {
        'prepare': household,
    },
    'aloi': {
        'prepare': aloi,
    },
    'census': {
        'prepare': census,
    },
    'celeba': {
        'prepare': celeba,
    },
    'blobs-2k-10-5': {
        'prepare': lambda: blobs(2_000, 10, 5),
    },
    'blobs-4k-10-5': {
        'prepare': lambda: blobs(4_000, 10, 5),
    },
    'blobs-8k-10-5': {
        'prepare': lambda: blobs(8_000, 10, 5),
    },
    'blobs-16k-10-5': {
        'prepare': lambda: blobs(8_000, 10, 5),
    },
    'blobs-32k-10-5': {
        'prepare': lambda: blobs(8_000, 10, 5),
    },
    'blobs-64k-10-5': {
        'prepare': lambda: blobs(8_000, 10, 5),
    },
    'blobs-100k-10-5': {
        'prepare': lambda: blobs(100_000, 10, 5),
    },
    'blobs-128k-10-5': {
        'prepare': lambda: blobs(8_000, 10, 5),
    },
}

