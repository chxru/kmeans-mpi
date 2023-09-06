import time
import numpy as np
import redis
from mpi4py import MPI
from constants import ITERATIONS, K, M
from kmeans.parallel import ParallelKMeans


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

prev_centroids = None


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
            data = []

            for i in range(per_process_data):
                message = r.lpop("queue")

                # message is a np.array in bytes format
                # we need to convert it back to np.array
                # before we can use it
                X = np.frombuffer(message, dtype=np.float64)
                data.append(X)

            # convert data to np.array
            data = np.array(data)
            print(f"Process {rank} consumed {data.shape[0]} data points")
            do_kmeans(data)

            time.sleep(3)
        else:
            print(f"Process {rank} waiting for data")
            time.sleep(1)


if __name__ == "__main__":
    listen_queue()
    # if rank == 0:
