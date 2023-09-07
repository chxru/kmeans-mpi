import csv
import os
import numpy as np
from mpi4py import MPI
from constants import ITERATIONS, K, M
from data.csv_utils import count_csv_rows, genereate_csv, load_csv_data
from kmeans.parallel import ParallelKMeans
from kmeans.sequential import SequentialKMeans
import time as t


# set seed for reproducibility
np.random.seed(1234)

# Data Loader Start Time
dt_StartTime = t.perf_counter()

# MPI stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# data source
filename = "./data/data_1000.csv"
if not os.path.exists(filename):
    genereate_csv(filename, 1000)

N = count_csv_rows(filename)


# split data into chunks
N_per_process = N // size
start_row = rank * N_per_process
end_row = N_per_process * (rank + 1) - 1

data = np.loadtxt(
    filename,
    delimiter=",",
    skiprows=start_row,
    max_rows=end_row - start_row + 1,
)

# Data loader End Time
dt_endTime = t.perf_counter()
print(f"Data Loader takes {dt_endTime - dt_StartTime:0.4f} seconds for processor {rank+1}")

# Time Stamp Begining for Parallel Processing Part
para_startTime = t.perf_counter()
# parallel part
parallel_kmeans = ParallelKMeans(X=data, K=K, D=M, iterations=ITERATIONS)
parallel_kmeans.fit(data)
# Time Stamp End for Parallel Processing Part
para_endTime = t.perf_counter()
print(f"Parallel Processing takes {para_endTime - para_startTime:0.4f} seconds for processor {rank+1}")
print(f"Whole Processing takes {para_endTime - dt_StartTime:0.4f} seconds for processor {rank+1}")

if rank == 0:
    # Data Loader Start time for Sequential and Scikit Processing 
    dta_startTime = t.perf_counter()
    X = load_csv_data(filename)
    # Data Loader End time for Sequential and Scikit Processing 
    dta_endTime = t.perf_counter()
    print(f"Alternate Data Loader takes {dta_endTime - dta_startTime:0.4f} seconds for processor {rank+1}")

    # Time Stamp Begining for Sequential Processing Part
    seq_startTime = t.perf_counter()
    # calculating kmeans sequentially
    sequential_kmeans = SequentialKMeans(K=K, D=M, data=X)
    sequential_kmeans.initial_centroids = parallel_kmeans.initial_centroids
    sequential_kmeans.fit(ITERATIONS)
    # Time Stamp Ending for Sequential Processing Part
    seq_endTime = t.perf_counter()
    print(f"Sequential Processing takes {para_endTime - para_startTime:0.4f} seconds for processor {rank+1}")

    # Time Stamp Begining for Scikit Processing Part
    sci_startTime = t.perf_counter()

    # validating results with Scikit learn library
    from sklearn.cluster import KMeans as SciktKMeans

    scikit_kmeans = SciktKMeans(
        n_init="auto",
        init=parallel_kmeans.initial_centroids,
        n_clusters=K,
        random_state=123,
        max_iter=ITERATIONS,
    ).fit(X)
    # Time Stamp end for Scikit Processing Part
    sci_endTime = t.perf_counter()
    print(f"Scikit Processing takes {sci_endTime - sci_startTime:0.4f} seconds for processor {rank+1}")

    # comparing results
    print("Sequential KMeans")
    print(sequential_kmeans.centroids)
    print("Parallel KMeans")
    print(parallel_kmeans.centroids)
    print("Scikit KMeans")
    print(scikit_kmeans.cluster_centers_)
    
