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
from benchmark.results import store_results, test_result_exists

def run_experiment(X: np.array, algo: BaseClustering):
    start = time.time()
    start_proc = time.process_time()
    algo.cluster(X)
    end = time.time()
    end_proc = time.process_time()
    return end - start - algo.get_overhead_time(), end_proc - start_proc - algo.get_overhead_time(), algo.retrieve_dendrogram(), algo.retrieve_milestones()


# def run_worker(dataset: str, queue: multiprocessing.Queue) -> None:
def run_worker(dataset: str, definition: Definition, n_run: int, overwrite: bool = True) -> None:
    runner = instantiate_algorithm(definition)

    # Immediately return iff file exists and the overwrite flag is false
    if not overwrite and test_result_exists(dataset, definition.algorithm, repr(runner)): return

    X = get_dataset(dataset)
    X = np.array(X["data"])
    # while not queue.empty():
        # definition = queue.get()

    time, time_proc, dendrogram, milestones = run_experiment(X, runner)
    attrs = {
        "time": time,
        "time_proc": time_proc,
        "ds": dataset,
        "run": n_run,
        "algo": definition.algorithm,
        "params": str(runner)
    }
    attrs.update(runner.get_additional())
    store_results(dataset, definition.algorithm, n_run, 
                    repr(runner), attrs, dendrogram, milestones)


def create_workers_and_execute(dataset: str, definitions: List[Definition], n_run=1, n_procs=1, overwrite=True) -> None:
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
    
    cpu_count = multiprocessing.cpu_count()
    if n_procs > cpu_count - 1:
       raise Exception(f"Parallelism larger than {cpu_count - 1}! (CPU count minus one)")

    timeout_hours = 10
    timeout_seconds = timeout_hours * 3600

    if n_procs == 1:
        # task_queue = multiprocessing.Queue()
        for run in definitions:
            # task_queue.put(run)

            try:
                workers = [multiprocessing.Process(target=run_worker, args=(dataset, run), kwargs=dict(n_run=n_run, overwrite=overwrite))]
                [worker.start() for worker in workers]
                [worker.join(timeout=timeout_seconds) for worker in workers] # Timeout of 10 hours

                for worker in workers:
                    if worker.is_alive():
                        print("Timeout reached. Terminating worker...")
                        worker.terminate()
                        worker.join()
                    else:
                        print("Worker completed within time.")
            finally:
                print("Terminating %d workers" % len(workers))
                [worker.terminate() for worker in workers]
    else:
        # Encapsulate the above code in a "nicer" way for a thread worker
        def pool_worker(run):
            try:
                worker = multiprocessing.Process(target=run_worker, args=(dataset, run), kwargs=dict(n_run=n_run, overwrite=overwrite))
                worker.start()
                worker.join(timeout=timeout_seconds) # Timeout of 10 hours
                if worker.is_alive(): raise TimeoutError()
                print("Worker completed within time.")
            except Exception as e:
                if worker.is_alive():
                    print("Timeout reached. Terminating worker...")
                    worker.terminate()
                    if worker.is_alive():
                        kill_result = os.system(f"kill {worker.pid}")
                        if kill_result: print(f"Warning: Failed to kill child process {worker.pid}")
                    else: worker.join()
                if type(e) != TimeoutError:
                    import traceback
                    traceback.print_exception(e)
        # Use a thread pool to have a managed number of executions running at any one time
        pool = multiprocessing.pool.ThreadPool(n_procs)
        pool.map(pool_worker, definitions)
        pool.close()


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

    parser.add_argument(
        '-r',
        '--run',
        type=int,
        default=1
    )

    parser.add_argument(
        '-o',
        '--overwrite',
        type=bool,
        default=True,
        help='whether or not to overwrite or skip existing results'
    )

    parser.add_argument(
        '-p',
        '--procs',
        type=int,
        default=1,
        help="the number of processes to use for experiments",
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

    create_workers_and_execute(args.dataset, definitions, n_run=args.run, n_procs=args.procs, overwrite=args.overwrite)