from mpi4py import MPI
import numpy as np
import time as t

from kmeans.base import BaseKMeans
from logger import log

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ParallelKMeans(BaseKMeans):
    @property
    def centroids(self):
        return self._centroids

    @property
    def initial_centroids(self):
        return self._initial_centroids

    @initial_centroids.setter
    def initial_centroids(self, value: np.ndarray):
        """
        DEBUG ONLY: set the initial centroids
        Useful for comparing the results of the parallel and serial implementations.
        Should call this method before calling fit().
        """
        self._centroids = value

    def __init__(
        self,
        K: int,
        D: int,
        X: np.ndarray,
        iterations: int,
        prev_centroids: np.ndarray = None,
    ) -> None:
        super().__init__()
        self._K = K
        self._iterations = iterations

        self._centroids = np.empty((K, D), dtype=np.float64)
        self._labels = None

        self._initialize_centroids(X, prev_centroids)
        self._initial_centroids = self._centroids.copy()

    def fit(self, X: np.ndarray):
        for _ in range(self._iterations):
            # calculate the distance between each data point and the centroids
            distance = self._calculate_euclidean_distance(X)

            # find the closest centroid for each data point
            self._labels = np.argmin(distance, axis=1)

            # update the centroids
            self._update_centroids(X)

    def predict(self, X):
        super().predict()
        distance = np.linalg.norm(X[:, None] - self._centroids, axis=2)
        return np.argmin(distance, axis=1)

    def _calculate_euclidean_distance(self, X: np.ndarray) -> np.ndarray:
        distance = np.linalg.norm(X[:, None] - self._centroids, axis=2)
        return distance

    def _initialize_centroids(
        self, X: np.ndarray, prev_centroids: np.ndarray = None
    ) -> None:
        if prev_centroids is not None:
            self._centroids = prev_centroids
            return

        if rank == 0:
            centroid_indices = np.random.choice(len(X), self._K, replace=False)
            self._centroids = X[centroid_indices.tolist()]

            # broadcast the centroids to all processes
            log("centroid_initialization", rank, t.perf_counter_ns())
            comm.bcast(self._centroids, root=0)
        else:
            self._centroids = comm.bcast(self._centroids, root=0)
            log("centroid_initialization", rank, t.perf_counter_ns())

    def _update_centroids(self, X: np.ndarray) -> None:
        for i in range(self._K):
            # if there is no data point assigned to this centroid, skip it
            if len(X[self._labels == i]) == 0:
                continue

            local_centroid = np.mean(X[self._labels == i], axis=0)

            tx = MPI.Wtime()
            local_centroid = comm.allreduce(local_centroid, op=MPI.SUM)
            tx = MPI.Wtime() - tx

            log("all_reduce", rank, tx * 1e9)

            self._centroids[i] = local_centroid / size
