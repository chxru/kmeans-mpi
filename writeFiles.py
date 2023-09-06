import random
import sys
import csv


def generate_data(num_rows):
    data = []
    for _ in range(num_rows):
        num1 = random.randint(1, 100)
        num2 = random.randint(1, 100)
        data.append((num1, num2))
    return data


def write_to_csv(data, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)


def main():
    if len(sys.argv) == 1:
        num_rows = 1000  # Default to 1000 if no argument is given
    elif len(sys.argv) == 2:
        try:
            num_rows = int(sys.argv[1])
        except ValueError:
            print("Invalid input. Please enter a valid integer for the number of rows.")
            sys.exit(1)
    else:
        print("Usage: python program.py [<number_of_rows>]")
        sys.exit(1)

    data = generate_data(num_rows)
    filename = f"data_{num_rows}.csv"

    write_to_csv(data, filename)
    print(f"Data has been written to {filename}.")


if __name__ == "__main__":
    main()
