import csv
import os
import random
import sys

import numpy as np


def genereate_csv(file_name: str, num_rows: int) -> None:
    """
    Generate a CSV file with the given number of rows.
    """
    # skip if file already exists
    if os.path.exists(file_name):
        return

    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        for _ in range(num_rows):
            # generate two floats between 0 and 1
            x = random.random()
            y = random.random()
            writer.writerow([x, y])


def count_csv_rows(csv_file: str) -> int:
    """
    Return the number of rows in a CSV file.
    """
    input_file = open(csv_file, "r+")
    reader_file = csv.reader(input_file)
    return len(list(reader_file))


def load_csv_data(filename):
    """
    Read a CSV file and return a numpy array.
    """
    data = []
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)


def read_csv_files_in_directory(directory):
    data_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", newline="") as csv_file:
                csv_reader = csv.reader(csv_file)
                data = [row for row in csv_reader]
            data_list.extend(data)
    numpy_array = np.array(data_list)
    return numpy_array.astype(np.float64)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python csv.py <num_rows>")
        sys.exit(1)

    # get number of rows from command line
    num_rows = int(sys.argv[1])

    # generate CSV files
    genereate_csv(f"data/data_{num_rows}.csv", num_rows)
