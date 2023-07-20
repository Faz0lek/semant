"""Sort lines in a text file using a language model.

    Date -- 03.07.2023
    Author -- Martin Kostelnik
"""

import logging
import argparse
import sys
import itertools
from typing import List
from tqdm import tqdm

import torch
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming

from semant.language_modelling.model import build_model, LanguageModel
from semant.language_modelling.tokenizer import build_tokenizer, LMTokenizer
from semant.language_modelling.utils import load_data
from semant.sorting.utils import compare_regions


BABICKA_PATH = r"/home/martin/semant/data/babicka/babicka.txt"
BABICKA_SHUFFLED_PATH = r"/home/martin/semant/data/babicka/babicka-shuffled.txt"


def parse_arguments():
    logging.info(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    # parser.add_argument("--file", required=True, type=str, help="Path to file to be sorted.")
    parser.add_argument("--model-path", required=True, type=str, help="Path to trained model.")
    parser.add_argument("--tokenizer-path", required=True, type=str, help="Path to trained tokenizer.")
    parser.add_argument("--save-path", required=True, type=str, help="Save result here.")

    args = parser.parse_args()
    return args


def sort_file(lines: List[str], tokenizer: LMTokenizer, model: LanguageModel) -> List[str]:
    n_lines = len(lines)

    # Create distance matrix
    logging.info("Creating distance matrix ...")
    DUMMY_DIST = 10.0
    distance_matrix = [[0.0] + n_lines * [DUMMY_DIST]] # Initialize with dummy point
    line_edges = [0.0]

    for sen1, sen2 in itertools.product(lines, repeat=2):
        sen1 = sen1.strip()
        sen2 = sen2.strip()

        if sen1 == sen2:
            line_edges.append(0.0)
        else:
            tokenized_seq = tokenizer(sen1, sen2)
            input_ids = tokenized_seq["input_ids"].to(model.device)
            token_type_ids = tokenized_seq["token_type_ids"].to(model.device)
            attention_mask = tokenized_seq["attention_mask"].to(model.device)

            with torch.no_grad():
                score = model(
                    input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                ).nsp_output.item()
            
            line_edges.append(score)

        if len(line_edges) == n_lines + 1:
            distance_matrix.append(line_edges)
            line_edges = [0.0]

    distance_matrix = np.array(distance_matrix)
    # As per python-tsp doc, to obtain open TSP version, set first column to 0
    distance_matrix[:, 0] = 0.0
    logging.info("Distance matrix created.")
    
    # with np.printoptions(precision=2, suppress=True):
    #     print(distance_matrix)

    # Solve TSP
    logging.info("Solving TSP ...")
    permutation, _ = solve_tsp_dynamic_programming(distance_matrix)
    permutation = [val - 1 for val in permutation[1:]]
    sorted_data = [lines[i] for i in permutation]

    return sorted_data


def main(args) -> None:
    logging.basicConfig(level=logging.WARNING, force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running inference on: {device}")

    # Load checkpoint
    logging.info("Loading checkpoint ...")
    checkpoint = torch.load(args.model_path)
    logging.info("Checkpoint loaded.")

    # Build tokenizer
    logging.info("Loading tokenizer ...")
    tokenizer = build_tokenizer(
        args.tokenizer_path,
        seq_len=checkpoint["seq_len"],
        fixed_sep=checkpoint["fixed_sep"],
        masking_prob=0.0
    )
    logging.info("Tokenizer loaded.")

    # Build model
    logging.info("Building model ...")
    model = build_model(
        czert=False,
        vocab_size=len(tokenizer),
        device=device,
        seq_len=checkpoint["seq_len"],
        out_features=516,
        mlm_level=2,
        sep=checkpoint["sep"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.mlm_head = None
    model = model.to(device)
    model.eval()
    logging.info("Model loaded.")

    # Load file
    logging.info("Loading input file ...")
    # raw_data = load_data(args.file)
    with open(BABICKA_PATH, "r") as f:
        babicka = f.read().split("\n\n")
    with open(BABICKA_SHUFFLED_PATH, "r") as f:
        babicka_shuffled = f.read().split("\n\n")
    logging.info("Input file loaded.")

    # Run sorting
    # sorted_data = sort_file(raw_data, tokenizer, model)
    total_hits = 0
    total_len = 0
    for region_s, region in tqdm(zip(babicka_shuffled, babicka)):
        region = region.split("\n")
        region_s = region_s.split("\n")
        sorted_data = sort_file(region_s, tokenizer, model)
        hits = compare_regions(region, sorted_data)

        total_hits += hits
        total_len += (len(region) - 1)
        # print("\n".join(region_s))
        # print()
        # print("\n".join(region))
        # print("\n\n\n")

    print(f"{total_hits}/{total_len}\t{(total_hits / total_len):.3f}")
    # Save result
    # with open(args.save_path, "w") as f:
    #     for line in sorted_data:
    #         print(line, end="", file=f)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
