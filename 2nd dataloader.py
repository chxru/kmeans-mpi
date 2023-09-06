from mpi4py import MPI
import numpy as np
from kmeans.parallel import ParallelKMeans
from kmeans.sequential import SequentialKMeans
import csv
import os
import pandas as pd

# parrelled part

np.random.seed(1234)

K = 3
M = 2
max_iter = 10

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

csv_directory = './data' 

csv_files = [os.path.join(csv_directory, file) for file in os.listdir(csv_directory) if file.endswith(".csv")]

total_row_count = 0
for file_path in csv_files:
    input_file = open(file_path,'r+')
    reader_file = csv.reader(input_file)
    total_row_count+=len(list(reader_file))

rows_per_process = total_row_count // size
start_row = rank * rows_per_process
end_row = start_row + rows_per_process

for file_path in csv_files:
    df = pd.read_csv(file_path, skiprows=range(1, start_row), nrows=rows_per_process)
    
N = total_row_count

# MPI stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# split data into chunks
N_per_process = N // size

# start_row = rank*N_per_process
# end_row = N_per_process*(rank+1)-1

data = df.to_numpy()
parallel_kmeans = ParallelKMeans(data=data, K=K, D=M)
parallel_kmeans.fit(max_iter)


if rank == 0:
    def read_csv_files_in_directory(directory):
        data_list = []
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', newline='') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    data = [row for row in csv_reader]
                data_list.extend(data)
        numpy_array = np.array(data_list)
        return numpy_array.astype(np.float64)

    X = read_csv_files_in_directory(csv_directory)
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
