from mpi4py import MPI
import numpy as np
from kmeans.parallel import ParallelKMeans
from kmeans.sequential import SequentialKMeans

np.random.seed(1234)

K = 3
N = 10000
M = 2
max_iter = 10

X = np.random.rand(N, M)

# MPI stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# split data into chunks
N_per_process = N // size

# calculating kmeans in parallely
data = X[rank * N_per_process : (rank + 1) * N_per_process]
parallel_kmeans = ParallelKMeans(data=data, K=K, D=M)
parallel_kmeans.fit(max_iter)


if rank == 0:
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
