from mpi4py import MPI
import numpy as np

from kmeans.base import BaseKMeans

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

    def __init__(self, K: int, D: int, data: np.ndarray) -> None:
        super().__init__()
        self._K = K
        self._data = data
        self._centroids = np.empty((K, D), dtype=np.float64)
        self._labels = None

        self._initialize_centroids(K)
        self._initial_centroids = self._centroids.copy()

    def fit(self, iterations: int):
        for _ in range(iterations):
            # calculate the distance between each data point and the centroids
            distance = self._calculate_euclidean_distance()

            # find the closest centroid for each data point
            self._labels = np.argmin(distance, axis=1)

            # update the centroids
            self._update_centroids()

    def predict(self, X):
        super().predict()
        distance = np.linalg.norm(X[:, None] - self._centroids, axis=2)
        return np.argmin(distance, axis=1)

    def _calculate_euclidean_distance(self) -> np.ndarray:
        distance = np.linalg.norm(self._data[:, None] - self._centroids, axis=2)
        return distance

    def _initialize_centroids(self, K: int) -> None:
        if rank == 0:
            centroid_indices = np.random.choice(len(self._data), K, replace=False)
            self._centroids = self._data[centroid_indices.tolist()]

            # broadcast the centroids to all processes
            comm.bcast(self._centroids, root=0)
        else:
            self._centroids = comm.bcast(self._centroids, root=0)

    def _update_centroids(self):
        for i in range(self._K):
            local_centroid = np.mean(self._data[self._labels == i], axis=0)
            local_centroid = comm.allreduce(local_centroid, op=MPI.SUM)
            self._centroids[i] = local_centroid / size
