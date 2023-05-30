"""Create test dataset from file with already processed dataset (from create_dataset.py).

    Input
    -----
        File in txt format with each line in format of <sen1>\t<sen2>\t<label>\n

    Output
    ------
        File in txt format where each line has a format <line1>\t<line2>\t<label>\n

    Date -- 30.05.2023
    Author -- Martin Kostelnik
"""

import argparse
import sys


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="Path to txt file with dataset.")
    parser.add_argument("--out", required=True, help="Path to output file with test dataset.")
    parser.add_argument("--out-filtered", required=True, help="Path to output file with original dataset without test samples.")

    parser.add_argument("--size", default=10000, type=int, help="Size of test dataset.")

    args = parser.parse_args()
    return args


def main(args):
    test_dataset = []
    filtered_dataset = []

    # [positive, positive_dot, negative, negative_dot]
    counts = [0] * 4
    limit = args.size // 4

    with open(args.data, "r") as f:
        for i, line in enumerate(f):
            if all(c == limit for c in counts):
                break

            sen1, sen2, label = line.split("\t")

            dot = sen1.strip().endswith(".")
            positive = int(label) == 0

            # negative_dot
            if counts[3] != limit and not positive and dot:
                counts[3] += 1
                test_dataset.append(line)

            # negative
            elif counts[2] != limit and not positive and not dot:
                counts[2] += 1
                test_dataset.append(line)

            # positive_dot
            elif counts[1] != limit and positive and dot:
                counts[1] += 1
                test_dataset.append(line)

            # positive
            elif counts[0] != limit and positive and not dot:
                counts[0] += 1
                test_dataset.append(line)
            
            else:
                filtered_dataset.append(line)

    with open(args.data, "r") as f:
        filtered_dataset.extend(f.readlines()[i:])
        
    with open(args.out_filtered, "w") as f:
        for line in filtered_dataset:
            print(line, file=f, end="")

    with open(args.out, "w") as f:
        for line in test_dataset:
            print(line, file=f, end="")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
