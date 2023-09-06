import os
import numpy as np
import csv

csv_directory ='./data'
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

    return numpy_array

X = read_csv_files_in_directory(csv_directory)

print(X.shape)