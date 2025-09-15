"""
Functionality to create datasets used in the evaluation.
"""
from glob import glob
import h5py
import os
import time
import numpy as np
import zipfile


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
    from benchmark.algorithms.scipy.module import SciPySingleLinkage
    print("Computing groundtruth...")
    start = time.time()
    d = X.shape[1]
    clustering = SciPySingleLinkage()
    clustering.cluster(X)
    end = time.time()
    print(f"Computing groundtruth took {(end - start):.2f}s.")
    return clustering.retrieve_dendrogram()

def write_output(X: np.ndarray, name: str, compute_gt=False, y=None):
    f = h5py.File(get_dataset_fn(name), "w")
    f.create_dataset("data", data=X)
    if y is not None: f.create_dataset("labels", data=y)
    if compute_gt:
        dendrogram = compute_groundtruth(X)
        f.create_dataset("dendrogram", data=dendrogram)
    f.close()

def monotonic_multisample(X, sample_sizes, seed=None):
    for sample_size in sample_sizes:
        if X.shape[0] < sample_size:
            print(f"Warning: Sample size ({sample_size}) should be less than the number of samples in X ({X.shape[0]}), using X.shape[0] instead.")
    if seed is None:
        perm = np.random.permutation(X.shape[0])
    else:
        perm = np.random.default_rng(seed=seed).permutation(X.shape[0])
    return [
        X[perm[:sample_size]]
        for sample_size in sample_sizes
    ]
def sample(X, sample_size, seed=None):
    return monotonic_multisample(X, sample_sizes=[sample_size], seed=seed)[0]

def mnist(sample_size=None, seed=None):
    from sklearn.datasets import fetch_openml
    
    if sample_size is not None:
        name = f"mnist-{sample_size//1_000}k"
    else:
        name = "mnist"
    
    if os.path.exists(get_dataset_fn(name)):    
        return
    
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    X = X.astype(np.float32)
    # X /= np.sum(X, axis=1, keepdims=True)
    
    if sample_size is None: sample_size = X.shape[0]
    X = sample(X, sample_size, seed=seed)

    write_output(X, name, y)


def pamap2(apply_pca=False, sample_size=None, seed=None):
    from sklearn.decomposition import PCA
    fn = "pamap2" if apply_pca else "pamap2-full"

    if sample_size is not None:
        name = f"{fn}-{sample_size//1_000}k"
    else:
        name = fn
    
    if os.path.exists(get_dataset_fn(name)):    
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
        
        X = X.astype(np.float32)
        if sample_size is None: sample_size = X.shape[0]
        X = sample(X, sample_size, seed=seed)

        write_output(X, name)

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


def aloi(sample_size=None, seed=None):
    
    if sample_size is not None:
        name = f"aloi-{sample_size//1_000}k"
    else:
        name = "aloi"
    
    if os.path.exists(get_dataset_fn(name)):    
        return
    
    src = "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/1_ALOI.npz"
    download(src, "aloi.npz")
    X = np.load("aloi.npz")['X']
    X = X.astype(np.float32)
    
    if sample_size is None: sample_size = X.shape[0]
    X = sample(X, sample_size, seed=seed)

    write_output(X, name)


def aloi77():
    name = "aloi-colorsim77"
    
    if os.path.exists(get_dataset_fn(name)):    
        return


def aloi733(sample_size=None, seed=None):
    if sample_size is not None:
        name = f"aloi733-{sample_size//1_000}k"
    else:
        name = "aloi733"
    
    if os.path.exists(get_dataset_fn(name)):    
        return
    
    try:
        path = 'data/aloi733.hdf5'
        with h5py.File(path, 'r') as f:
            X = f['data'][:]
    except:
        print('fejl i path')

    X = X.astype(np.float32)
    if sample_size is None: sample_size = X.shape[0]
    X = sample(X, sample_size, seed=seed)

    write_output(np.array(X), name)
    

def census(sample_size=None, seed=None):
    if sample_size is not None:
        name = f"census-{sample_size//1_000}k"
    else:
        name = "census"
    
    if os.path.exists(get_dataset_fn(name)):
        return

    src = "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/9_census.npz"
    download(src, "census.npz")
    X = np.load("census.npz")['X']

    X = X.astype(np.float32)
    if sample_size is None: sample_size = X.shape[0]
    X = sample(X, sample_size, seed=seed)

    write_output(np.array(X), name)


def celeba(sample_size=None, seed=None):
    if sample_size is not None:
        name = f"celeba-{sample_size//1_000}k"
    else:
        name = "celeba"
    
    if os.path.exists(get_dataset_fn(name)):
        return
    
    src = "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/8_celeba.npz"
    download(src, "celeba.npz")
    X = np.load("celeba.npz")['X']

    X = X.astype(np.float32)
    if sample_size is None: sample_size = X.shape[0]
    X = sample(X, sample_size, seed=seed)

    write_output(np.array(X), name)


def blobs(n, dim, centers, seed=42, noise_seed=1, n_noise=None):
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler

    name = f"blobs-{n // 1_000}k-{dim}-{centers}"
    if n_noise is not None: name = f"{name}-{n_noise // 1_000}k"

    if os.path.exists(get_dataset_fn(name)):
        return

    X, y = make_blobs(n, dim, centers=centers, random_state=seed)
    if n_noise is not None:
        # Ensure the same seed for all noise samples to have
        # smaller noise samples be subsets of larger noise
        # samples.
        X, y = add_noise(X, n_noise, y=y, seed=noise_seed)
    X = X.astype(np.float32)
    write_output(np.array(X), name, y=y)

def add_noise(X, n_noise, y=None, seed=None):
    lo, hi = np.min(X,axis=0), np.max(X,axis=0)
    if seed is not None: np.random.seed(seed)
    noise = np.random.sample((n_noise, X.shape[1])) * (hi-lo) + lo
    X_noised = np.concatenate([X, noise], axis=0)
    if y is None:
        return X_noised
    y_noised = np.concatenate([y, np.full(noise.shape[0], -1)])
    return X_noised, y_noised


DATASETS = {
    **{
        f'mnist-{i}k': {
            'prepare': (lambda i: (lambda: mnist(i*1_000, seed=0)))(i),
        }
        for i in (1+np.arange(7))*10
    },
    **{
        f'mnist-{i}k': {
            'prepare': (lambda i: (lambda: mnist(i*1_000, seed=0)))(i),
        }
        for i in [8, 16, 23, 31, 39, 47, 54, 62, 70]
    },
    **{
        f'mnist-{i}k': {
            'prepare': (lambda i: (lambda: mnist(i*1_000, seed=0)))(i),
        }
        for i in [12, 15, 19, 24, 29, 37, 45, 56]
    },
    **{
        f'aloi-{i}k': {
            'prepare': (lambda i: (lambda: aloi(i*1_000, seed=0)))(i),
        }
        for i in (1+np.arange(11))*10
    },
    **{
        f'aloi-{i}k': {
            'prepare': (lambda i: (lambda: aloi(i*1_000, seed=0)))(i),
        }
        for i in [11, 13, 15, 16, 17, 23, 25, 27, 32, 33]
    },
    **{
        f'aloi733-{i}k': {
            'prepare': (lambda i: (lambda: aloi733(i*1_000, seed=10)))(i),
        }
        for i in [50, 55, 60, 65, 71, 77, 85, 92, 101, 110]
    },
    **{
        f'aloi733-{i}k': {
            'prepare': (lambda i: (lambda: aloi733(i*1_000, seed=10)))(i),
        }
        for i in [12, 25, 37, 49, 62, 74, 86, 99, 111]
    },
    'aloi733': {
            'prepare': (lambda: aloi733()),
    },
    **{
        f'celeba-{i}k': {
            'prepare': (lambda i: (lambda: celeba(i*1_000, seed=0)))(i),
        }
        for i in [23, 45, 68, 90, 113, 135, 158, 180, 203]
    },
    **{
        f'census-{i}k': {
            'prepare': (lambda i: (lambda: census(i*1_000, seed=0)))(i),
        }
        for i in [28, 56, 83, 111, 139, 167, 194, 222, 250]
    },
    **{
        f'pamap2-{i}k': {
            'prepare': (lambda i: (lambda: pamap2(True, i*1_000, seed=0)))(i),
        }
        for i in (1+np.arange(25))*10
    },    
    **{
        f'pamap2-{i}k': {
            'prepare': (lambda i: (lambda: pamap2(True, i*1_000, seed=0)))(i),
        }
        for i in [50, 65, 83, 108, 139, 180, 232, 300, 387, 500]
    },
    'pamap2': {
        'prepare': lambda: pamap2(True),
    },
    'pamap2-full': {
        'prepare': lambda: pamap2(False),
    },
    **{
        f.__name__: {'prepare': f}
        for f in [mnist, aloi, household, census, celeba]
    },
    **{
        f'blobs-{i}k-{dim}-{centers}': {
            'prepare': (lambda i, dim, centers: (lambda: blobs(i*1_000, dim, centers)))(i, dim, centers),
        }
        for i in [2, 4, 8, 16, 32, 64, 100, 128, 256, 512]
        for dim in [5, 10]
        for centers in [5, 20, 50, 100]
    },
    **{
        f'blobs-{i}k-{dim}-{centers}-{noise // 1_000}k': {
            'prepare': (lambda i, dim, centers, noise: (lambda: blobs(i*1_000, dim, centers, n_noise=noise)))(i, dim, centers, noise),
        }
        for i in [2, 4, 8, 16, 32, 64, 100, 128]
        for dim in [5, 10]
        for centers in [5, 20, 50, 100]
        for noise in [10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 75_000, 80_000]
    },
}

