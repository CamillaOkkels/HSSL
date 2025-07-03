import json
import os
import re
import traceback
import numpy as np
from typing import Any, Optional, Set, Tuple, Iterator
import h5py


def build_result_filepath(dataset_name: Optional[str] = None, 
                          algorithm: Optional[any] = None,
                          run: Optional[any] = None,
                          arguments: Optional[Any] = None) -> str:
    d = ["results"]
    if dataset_name:
        d.append(dataset_name)
    if algorithm:
        d.append(algorithm)
        if run:
            d.append(f"run={run}")
        #data = definition.arguments + query_arguments
        #d.append(re.sub(r"\W+", "_", json.dumps(data, sort_keys=True)).strip("_") + ".hdf5")
            if arguments:
                d.append(arguments + ".hdf5")
            else:
                d.append("run.hdf5")
    return os.path.join(*d)

def test_result_exists(dataset_name: str, algorithm: str, arguments: str) -> bool:
    filename = build_result_filepath(dataset_name, algorithm, arguments)
    return os.path.isfile(filename)

def store_results(dataset_name: str, algorithm: str, run: int, 
            arguments: str, attrs, children, milestones):
    filename = build_result_filepath(dataset_name, algorithm, run, arguments)
    directory, _ = os.path.split(filename)

    print(f"storing in {filename}")

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with h5py.File(filename, "w") as f:
        for k, v in attrs.items():
            print(k, v)
            f.attrs[k] = v
        f.create_dataset("dendrogram", data=children)
        f.create_dataset("milestones", data=np.bytes_(json.dumps(milestones)))
    


def load_all_results(dataset: Optional[str] = None, prefix: str = ".") -> Iterator[Tuple[h5py.File]]:
    for root, _, files in os.walk(os.path.join(prefix, build_result_filepath(dataset))):
        for filename in files:
            if os.path.splitext(filename)[-1] != ".hdf5":
                continue
            try:
                yield h5py.File(os.path.join(root, filename), "r")
            except Exception:
                print(f"Was unable to read {filename}")
                traceback.print_exc()


def get_unique_algorithms() -> Set[str]:
    """
    Retrieves unique algorithm names from the results.

    Returns:
        set: A set of unique algorithm names.
    """
    algorithms = set()
    for properties, _ in load_all_results():
        algorithms.add(properties["algo"])
    return algorithms