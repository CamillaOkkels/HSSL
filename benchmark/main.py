import argparse
import numpy as np
import time
import multiprocessing.pool
import os
import threading
import json


from typing import List

from benchmark.datasets import DATASETS, get_dataset
from benchmark.definitions import instantiate_algorithm, get_definitions, list_algorithms, Definition
from benchmark.algorithms.base.module import BaseClustering
from benchmark.results import store_results

def run_experiment(X: np.array, algo: BaseClustering):
    start = time.time()
    algo.cluster(X)
    end = time.time()
    return end - start, algo.retrieve_dendrogram()


def run_worker(dataset: str, queue: multiprocessing.Queue) -> None:
    X = get_dataset(dataset)
    X = np.array(X["data"])
    while not queue.empty():
        definition = queue.get()


        runner = instantiate_algorithm(definition)

        time, dendrogram = run_experiment(X, runner)
        attrs = {
            "time": time,
            "ds": dataset,
            "algo": definition.algorithm,
            "params": str(runner)
        }
        attrs.update(runner.get_additional())
        store_results(dataset, definition.algorithm, 
                      repr(runner), attrs, dendrogram)

def create_workers_and_execute(dataset: str, definitions: List[Definition]) -> None:
    """
    Manages the creation, execution, and termination of worker processes based on provided arguments.

    Args:
        definitions (List[Definition]): List of algorithm definitions to be processed.
        args (argparse.Namespace): User provided arguments for running workers.

    Raises:
        Exception: If the level of parallelism exceeds the available CPU count or if batch mode is on with more than
                   one worker.
    """
    #cpu_count = multiprocessing.cpu_count()
    #if args.parallelism > cpu_count - 1:
    #    raise Exception(f"Parallelism larger than {cpu_count - 1}! (CPU count minus one)")

    # if args.batch and args.parallelism > 1:
    #     raise Exception(
    #         f"Batch mode uses all available CPU resources, --parallelism should be set to 1. (Was: {args.parallelism})"
    #     )

    task_queue = multiprocessing.Queue()
    for run in definitions:
        task_queue.put(run)

    try:
        workers = [multiprocessing.Process(target=run_worker, args=(dataset, task_queue))]
        [worker.start() for worker in workers]
        [worker.join() for worker in workers]
    finally:
        print("Terminating %d workers" % len(workers))
        [worker.terminate() for worker in workers]

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--dataset',
        metavar='NAME',
        help='the dataset to cluster',
        default='mnist',
        choices=DATASETS.keys()
    )

    parser.add_argument(
        '--algorithm',
    )

    parser.add_argument(
        '--list-algorithms',
        action='store_true',
        help="list available algorithms"
    )

    parser.add_argument(
        '--prepare',
        action='store_true',
        help='only prepare the dataset'
    )

    args = parser.parse_args()


    if args.list_algorithms:
        list_algorithms()
        exit(0)

    definitions = list(get_definitions())

    if args.algorithm:
        definitions = [d for d in definitions if d.algorithm == args.algorithm]


    # get definitions here

    ds = DATASETS[args.dataset]
    print(f"preparing {args.dataset}")
    ds['prepare']()

    if args.prepare:
        exit(0)

    create_workers_and_execute(args.dataset, definitions)