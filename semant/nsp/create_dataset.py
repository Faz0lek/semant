"""Create dataset from txt file with preprocess data (from preprocess_data.py).

    Input
    -----
        File in txt format with each text region separated by \n\n

    Output
    ------
        File in txt format where each line has a format <line1>\t<line2>\t<label>\n

    Date -- 29.05.2023
    Author -- Martin Kostelnik
"""

import argparse
import sys
from itertools import pairwise, permutations
from random import sample


POSITIVE_LABEL = 0
NEGATIVE_LABEL = 1


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="Path to txt file with preprocessed data.")
    parser.add_argument("--out", required=True, help="Path to output file.")
    parser.add_argument("--balanced", action="store_true", help="Whether the dataset should be balanced. If not using balanced dataset, every possible wrong combination of lines will be generated from a given region.")
    parser.add_argument("--header", action="store_true", help="Whether regions in preprocessed file contain header. If set, removes the header.")

    args = parser.parse_args()
    return args


def main(args):
    with open(args.data, "r") as data_f:
        data = data_f.read()
    
    regions = data.split("\n\n")

    with open(args.out, "w") as out_f:
        for region in regions:
            lines = region.split("\n")
            region_samples = ""

            # Filter out header
            if args.header:
                lines = lines[1:]
                if len(lines) < 2:
                    continue

            # Generate positive samples
            n_samples = 0
            for pair in pairwise(lines):
                region_samples += f"{pair[0]}\t{pair[1]}\t{POSITIVE_LABEL}\n"
                n_samples += 1

            # Generate negative samples
            idxs = [(idx1, idx2) for (idx1, idx2) in permutations(range(0, len(lines)), 2) if idx2 != idx1 + 1]
            
            # Sample only the same amount of negative samples as we have positive samples
            if args.balanced:
                idxs = sample(idxs, n_samples)

            for idx1, idx2 in idxs:
                region_samples += f"{lines[idx1]}\t{lines[idx2]}\t{NEGATIVE_LABEL}\n"

            print(region_samples, file=out_f, end="")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
