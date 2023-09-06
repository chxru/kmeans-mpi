from mpi4py import MPI
import numpy as np
from kmeans.parallel import ParallelKMeans
from kmeans.sequential import SequentialKMeans
import csv
import writeFiles
import os

# Parrellel Part

np.random.seed(1234)

K = 3
M = 2
max_iter = 10

# counting the number of rows

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def count_csv_rows(csv_file):
    if rank == 0:
        if not os.path.exists(csv_file):
            writeFiles.main()

    input_file = open(csv_file, "r+")
    reader_file = csv.reader(input_file)
    return len(list(reader_file))


filename = "data_1000.csv"

N = count_csv_rows(filename)

# MPI stuff

# split data into chunks
N_per_process = N // size
start_row = rank * N_per_process
end_row = N_per_process * (rank + 1) - 1

data = np.loadtxt(
    filename, delimiter=",", skiprows=start_row, max_rows=end_row - start_row + 1
)
parallel_kmeans = ParallelKMeans(data=data, K=K, D=M, iterations=max_iter)
parallel_kmeans.fit(data)

# series part
if rank == 0:

    def load_data(filename):
        data = []
        with open(filename, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data.append([float(x) for x in row])
        return np.array(data)

    X = load_data(filename)
    # calculating kmeans sequentially
    sequential_kmeans = SequentialKMeans(K=K, D=M, data=X)
    sequential_kmeans.initial_centroids = parallel_kmeans.initial_centroids
    sequential_kmeans.fit(max_iter)

    # validating results with Scikit learn library
    from sklearn.cluster import KMeans as SciktKMeans

    scikit_kmeans = SciktKMeans(
        n_init="auto",
        init=parallel_kmeans.initial_centroids,
        n_clusters=K,
        random_state=123,
        max_iter=max_iter,
    ).fit(X)

    # comparing results
    print("Sequential KMeans")
    print(sequential_kmeans.centroids)
    print("Parallel KMeans")
    print(parallel_kmeans.centroids)
    print("Scikit KMeans")
    print(scikit_kmeans.cluster_centers_)
