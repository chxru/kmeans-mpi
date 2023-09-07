from mpi4py import MPI
import numpy as np
from constants import ITERATIONS, K, M
from data.csv_utils import multifile_loadbalancer, read_csv_files_in_directory
from kmeans.parallel import ParallelKMeans
from kmeans.sequential import SequentialKMeans
import os
import pandas as pd

np.random.seed(1234)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


csv_directory = "./data"
if rank == 0:
    balance_dict = multifile_loadbalancer(csv_directory, size)

    # broadcast balance_dict to all processes
    comm.bcast(balance_dict, root=0)

else:
    balance_dict = comm.bcast(None, root=0)

# get the files that this process will read
files_to_read = balance_dict[rank]

# create empty dataframe
df = None

for file in files_to_read:
    file_path = os.path.join(csv_directory, file["file"])

    if df is None:
        df = pd.read_csv(
            file_path,
            skiprows=range(1, file["start"]),
            nrows=file["end"],
            names=["x", "y"],
        )
    else:
        temp = pd.read_csv(
            file_path,
            skiprows=range(1, file["start"]),
            nrows=file["end"],
            names=["x", "y"],
        )

        df = pd.concat([df, temp], axis=0, ignore_index=True)

data = df.to_numpy()

parallel_kmeans = ParallelKMeans(X=data, K=K, D=M, iterations=ITERATIONS)
parallel_kmeans.fit(data)


if rank == 0:
    X = read_csv_files_in_directory(csv_directory)

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
