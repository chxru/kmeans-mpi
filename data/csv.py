import random
import sys
import csv

import numpy as np


def genereate_csv(file_name: str, num_rows: int) -> None:
    """
    Generate a CSV file with the given number of rows.
    """
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
