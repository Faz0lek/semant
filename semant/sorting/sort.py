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
import random

import torch
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming

from semant.language_modelling.model import build_model, LanguageModel
from semant.language_modelling.tokenizer import build_tokenizer, LMTokenizer
from semant.language_modelling.utils import load_data
from semant.sorting.utils import compare_regions, split_into_regions

from time import perf_counter


BABICKA_PATH = r"/home/martin/semant/data/babicka/babicka.txt"
BABICKA_SHUFFLED_PATH = r"/home/martin/semant/data/babicka/babicka-shuffled.txt"
BABICKA_EASY_PATH = r"/home/martin/semant/data/babicka/babicka-easy.txt"
BABICKA_10_PATH = r"/home/martin/semant/data/babicka/babicka_10.txt"


def parse_arguments():
    logging.info(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    # parser.add_argument("--file", required=True, type=str, help="Path to file to be sorted.")
    parser.add_argument("--model-path", required=True, type=str, help="Path to trained model.")
    parser.add_argument("--tokenizer-path", required=True, type=str, help="Path to trained tokenizer.")
    parser.add_argument("--save-path", required=True, type=str, help="Save result here.")

    args = parser.parse_args()
    return args


def sort_file_tsp_local(lines: List[str], tokenizer: LMTokenizer, model: LanguageModel) -> List[str]:
    """Assumptions:
           1. Errors are local, line that is on index 14 (out of 15) does not correctly belong on index 2

    """
    n_lines = len(lines)
    idx_offsets = [-2, -1, 1, 2]

    # Create distance matrix
    logging.info("Creating distance matrix ...")
    start = perf_counter()
    DUMMY_DIST = 10.0
    distance_matrix = [[0.0] + n_lines * [DUMMY_DIST]] # Initialize with dummy point
    line_edges = [0.0] + [1.0] * n_lines

    for idx, sen1 in enumerate(lines):
        sen1 = sen1.strip()
        line_edges[idx+1] = 0.0
        
        for offset in idx_offsets:
            idx_ = idx + offset
            if idx_ < 0:
                continue

            try:
                sen2 = lines[idx_].strip()
            except IndexError:
                continue

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

            line_edges[idx_ + 1] = score

        distance_matrix.append(line_edges)
        line_edges = [0.0] + [1.0] * n_lines

    distance_matrix = np.array(distance_matrix)
    # As per python-tsp doc, to obtain open TSP version, set first column to 0
    distance_matrix[:, 0] = 0.0
    end = perf_counter()
    dmt = end - start
    # print(distance_matrix.shape)
    # with np.printoptions(precision=2, suppress=True):
    #     print(distance_matrix)

    # Solve TSP
    logging.info("Solving TSP ...")
    start = perf_counter()
    permutation, _ = solve_tsp_dynamic_programming(distance_matrix)
    permutation = [val - 1 for val in permutation[1:]]
    sorted_data = [lines[i] for i in permutation]
    end = perf_counter()
    tspt = end - start
    
    print(f"Distance matrix: {dmt:.2f}\tTSP: {tspt:.2f}")
    return sorted_data


def sort_file(lines: List[str], tokenizer: LMTokenizer, model: LanguageModel) -> List[str]:
    n_lines = len(lines)

    # Create distance matrix
    logging.info("Creating distance matrix ...")
    DUMMY_DIST = 10.0
    distance_matrix = [[0.0] + n_lines * [DUMMY_DIST]] # Initialize with dummy point
    # distance_matrix = [] 
    line_edges = [0.0]
    # line_edges = []

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
            # line_edges = []

    distance_matrix = np.array(distance_matrix)
    # As per python-tsp doc, to obtain open TSP version, set first column to 0
    distance_matrix[:, 0] = 0.0
    logging.info("Distance matrix created.")
    
    # with np.printoptions(precision=2, suppress=True):
    #     print(distance_matrix)

    # Solve TSP
    logging.info("Solving TSP ...")
    start = perf_counter()
    permutation, _ = solve_tsp_dynamic_programming(distance_matrix)
    # permutation, _ = solve_tsp_simulated_annealing(distance_matrix)
    end = perf_counter()
    permutation = [val - 1 for val in permutation[1:]]
    sorted_data = [lines[i] for i in permutation]

    # print(f"TSP: {(end-start):.2f}")
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
        czert=checkpoint["czert"],
        vocab_size=len(tokenizer),
        device=device,
        seq_len=checkpoint["seq_len"],
        out_features=checkpoint["features"],
        mlm_level=0,
        sep=checkpoint["sep"],
    )
    model.bert.load_state_dict(checkpoint["bert_state_dict"])
    model.nsp_head.load_state_dict(checkpoint["nsp_head_state_dict"])
    model = model.to(device)
    model.eval()
    logging.info("Model loaded.")

    # Load file
    logging.info("Loading input file ...")
    # raw_data = load_data(args.file)

    REGION_SIZE = 15
    with open(BABICKA_PATH, "r") as f:
        babicka = f.read()

    babicka_shuffled = split_into_regions(babicka, REGION_SIZE, True)    
    babicka = split_into_regions(babicka, REGION_SIZE)
    logging.info("Input file loaded.")

    # Run sorting
    total_hits = 0
    total_len = 0
    # for region_s, region in tqdm(zip(babicka_shuffled, babicka)):
    for region_s, region in tqdm(zip(babicka_shuffled, babicka)):
        region = region.split("\n")
        region_s = region_s.split("\n")
        sorted_data = sort_file(region_s, tokenizer, model)
        # sorted_data = sort_file_tsp_local(region_s, tokenizer, model)
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
