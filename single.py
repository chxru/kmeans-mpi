import csv
import os
import numpy as np
from mpi4py import MPI
from constants import ITERATIONS, K, M
from data.generate_csv import count_csv_rows, genereate_csv, load_csv_data
from kmeans.parallel import ParallelKMeans
from kmeans.sequential import SequentialKMeans

# set seed for reproducibility
np.random.seed(1234)

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

# parallel part
parallel_kmeans = ParallelKMeans(X=data, K=K, D=M, iterations=ITERATIONS)
parallel_kmeans.fit(data)

if rank == 0:
    X = load_csv_data(filename)

    # calculating kmeans sequentially
    sequential_kmeans = SequentialKMeans(K=K, D=M, data=X)
    sequential_kmeans.initial_centroids = parallel_kmeans.initial_centroids
    sequential_kmeans.fit(ITERATIONS)

    # validating results with Scikit learn library
    from sklearn.cluster import KMeans as SciktKMeans

    scikit_kmeans = SciktKMeans(
        n_init="auto",
        init=parallel_kmeans.initial_centroids,
        n_clusters=K,
        random_state=123,
        max_iter=ITERATIONS,
    ).fit(X)

    # comparing results
    print("Sequential KMeans")
    print(sequential_kmeans.centroids)
    print("Parallel KMeans")
    print(parallel_kmeans.centroids)
    print("Scikit KMeans")
    print(scikit_kmeans.cluster_centers_)
