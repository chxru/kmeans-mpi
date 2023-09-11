import time
import numpy as np
import redis
from mpi4py import MPI
from constants import ITERATIONS, K, M
from data.utils import reduce_data_points
from kmeans.parallel import ParallelKMeans
import time as t


# MPI stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# redis stuff
r = redis.Redis(host="localhost", port=6379, db=0)
per_process_data = 10
bulk_size = (
    size * per_process_data
)  # cheaky way to make sure each process gets equal amount of data

prev_centroids: np.ndarray = (
    None  # this is used to save the centroids from the previous iteration
)
saved_data: np.ndarray = None  # this is used to save the previous data points


def do_kmeans(data: np.ndarray) -> None:
    global prev_centroids

    parallel_kmeans = ParallelKMeans(
        X=data,
        K=K,
        D=M,
        iterations=ITERATIONS,
        prev_centroids=prev_centroids if prev_centroids is not None else None,
    )
    parallel_kmeans.fit(data)

    # save the centroids for the next iteration
    prev_centroids = parallel_kmeans.centroids

    if rank == 0:
        print(parallel_kmeans.centroids)


def listen_queue() -> None:
    while True:
        queue_length = r.llen("queue")

        if queue_length > bulk_size:
            # Queued Process Starting Time
            que_startTime = t.perf_counter()

            data = []

            for _ in range(per_process_data):
                message = r.lpop("queue")

                # message is a np.array in bytes format
                # we need to convert it back to np.array
                # before we can use it
                X = np.frombuffer(message, dtype=np.float64)
                data.append(X)

            print(
                f"Process {rank} consumed {len(data)} data points in {t.perf_counter() - que_startTime:0.4f} seconds"
            )

            # convert data to np.array
            data = np.array(data)

            global saved_data

            # if data hsa been previously saved, concatenate it with the new data
            if saved_data is not None:
                data = np.concatenate((saved_data, data))

            kmeans_start_time = t.perf_counter()
            do_kmeans(data)

            print(
                f"Process {rank} kmeans took {t.perf_counter() - kmeans_start_time:0.4f} seconds"
            )

            # reduce the number of data points, to preserve memory
            points = reduce_data_points(data, rank)

            # save the reduced data points for the next iteration
            saved_data = points

            # uncomment this part if you want to see the data points
            # import matplotlib.pyplot as plt

            # plt.scatter(data[:, 0], data[:, 1])
            # plt.scatter(points[:, 0], points[:, 1], c="red")
            # plt.show()

            time.sleep(3)
        else:
            print(f"Process {rank} waiting for data")
            time.sleep(1)


if __name__ == "__main__":
    listen_queue()
