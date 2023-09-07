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
    with open(csv_file, mode="r") as file:
        return sum(1 for _ in file)


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


def multifile_loadbalancer(path: str, n: int) -> dict:
    # os listdir and filter with .csv
    files = [f for f in os.listdir(path) if f.endswith(".csv")]

    # get total number of rows of each files
    file_dict = {}
    for file in files:
        file_dict[file] = count_csv_rows(os.path.join(path, file))

    # get total number of rows
    total_rows = sum(file_dict.values())

    # get number of rows per process
    rows_per_process = total_rows // n

    offset = 0
    done_files = []
    final_dict = {}

    for i in range(n):
        final_dict[i] = []

    # iterate through each process
    for i in range(n):
        missing_rows = rows_per_process

        for file in files:
            if file not in done_files:
                # check whether the file has enough rows
                if offset + missing_rows <= file_dict[file]:
                    # if so, start with the previous offset until the missing rows
                    final_dict[i].append(
                        {
                            "file": file,
                            "start": offset,
                            "end": offset + missing_rows,
                        }
                    )

                    if offset + missing_rows == file_dict[file]:
                        # has reached end of the file
                        done_files.append(file)
                        offset = 0
                    else:
                        offset += missing_rows

                    # break if necessary rows per process has been reached
                    break
                else:
                    final_dict[i].append(
                        {
                            "file": file,
                            "start": offset,
                            "end": file_dict[file],
                        }
                    )
                    done_files.append(file)

                    missing_rows = file_dict[file] - offset
                    offset = 0
                    if missing_rows < 0:
                        missing_rows = 0
                        break

    for i in final_dict:
        print(f"Process {i}: {final_dict[i]}")

    return final_dict


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python csv.py <num_rows>")
        sys.exit(1)

    # get number of rows from command line
    num_rows = int(sys.argv[1])

    # generate CSV files
    genereate_csv(f"data/data_{num_rows}.csv", num_rows)
