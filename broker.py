import time
import numpy as np
import redis
from mpi4py import MPI
from constants import ITERATIONS, K, M
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

prev_centroids = None


def do_kmeans(data: np.ndarray) -> None:
    # do_Kmeans Start Time 
    #km_startTime = t.perf_counter()

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

    # do_Kmeans End Time 
    #km_endTime = t.perf_counter()
    #print(f"kMeans process takes {km_endTime - km_startTime:0.4f} seconds for processor {rank+1}")


def listen_queue() -> None:
    while True:
        # Queued Process Starting Time 
        que_startTime  = t.perf_counter()
        
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
        
        # Queued Process Ending Time 
        que_endTime  = t.perf_counter()
        print(f"Queued process takes {que_endTime - que_startTime:0.4f} seconds for processor {rank+1}")


if __name__ == "__main__":
    listen_queue()
    # if rank == 0:
