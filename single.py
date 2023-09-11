import numpy as np
from mpi4py import MPI
from constants import ITERATIONS, K, M
from data.csv_utils import count_csv_rows, genereate_csv, load_csv_data
from kmeans.parallel import ParallelKMeans
from kmeans.sequential import SequentialKMeans
import time as t

from logger import log


def main(rows: int):
    # data source
    filename = f"./data/data_{rows}.csv"
    genereate_csv(filename, rows)

    # set seed for reproducibility
    np.random.seed(1234)

    # MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    """
    DATA LOADING
    """
    dl_start_timer = t.perf_counter_ns()  # Data Loader Start Time

    # split data into chunks
    N = count_csv_rows(filename)
    N_per_process = N // size
    start_row = rank * N_per_process
    end_row = N_per_process * (rank + 1) - 1

    data = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=start_row,
        max_rows=end_row - start_row + 1,
    )

    log("loader", rank, t.perf_counter_ns() - dl_start_timer)

    """
    PROCESSING
    """

    process_start_timer = t.perf_counter_ns()  # Parallel Processing Start Time

    parallel_kmeans = ParallelKMeans(X=data, K=K, D=M, iterations=ITERATIONS)
    parallel_kmeans.fit(data)

    log("processing", rank, t.perf_counter_ns() - process_start_timer)


if __name__ == "__main__":
    import sys

    # check for command line argument
    if len(sys.argv) != 2:
        print("Usage: python3 single.py <n>")
        exit(1)

    # get n from command line
    n = int(sys.argv[1])
    if n < 1:
        print("n must be greater than 0")
        exit(1)

    main(n)
