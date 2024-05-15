import argparse
import os
import sys
import logging
from typing import List

import numpy as np
import torch
from python_tsp.exact import solve_tsp_dynamic_programming

from semant.language_modelling.tokenizer import LMTokenizer, build_tokenizer
from semant.language_modelling.model import LanguageModel, build_model


FOLDER = r"/home/martin/semant/data/periodicals/periodical-sample"


def parse_arguments():
    logging.info(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--file", required=True, type=str, help="Path to file to be sorted.")
    parser.add_argument("--model-path", required=True, type=str, help="Path to trained model.")
    parser.add_argument("--tokenizer-path", required=True, type=str, help="Path to trained tokenizer.")
    # parser.add_argument("--save-path", required=True, type=str, help="Save result here.")
    args = parser.parse_args()
    return args


def create_distance_matrix(
    end_lines: List[str],
    start_lines: List[str],
    tokenizer: LMTokenizer,
    model: LanguageModel
) -> np.ndarray:
    assert len(end_lines) == len(start_lines)
    n_lines = len(end_lines)

    # Create distance matrix
    DUMMY_DIST = 10.0
    distance_matrix = [[0.0] + n_lines * [DUMMY_DIST]] # Initialize with dummy point
    # distance_matrix = []
    line_edges = [0.0]
    # line_edges = []

    for end_i, end_line in enumerate(end_lines):
        for start_i, start_line in enumerate(start_lines):
            sen1 = end_line.strip()
            sen2 = start_line.strip()

            if end_i == start_i:
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
    
    # with np.printoptions(precision=2, suppress=True):
    #    print(distance_matrix)

    return distance_matrix


def sort_file_tsp(
    distance_matrix: np.ndarray
) -> List[int]:
    # Solve TSP
    permutation, _ = solve_tsp_dynamic_programming(distance_matrix)
    permutation = [val - 1 for val in permutation[1:]]

    return permutation


def main(args):
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

    filenames = os.listdir(FOLDER)
    for filename in filenames:
        if not filename.endswith("txt"):
            continue
        print(f"Sorting {filename}")

        with open(os.path.join(FOLDER, filename), "r") as f:
            regions = f.read().split("\n\n")

        start_lines = []
        end_lines = []

        if len(regions) > 16:
            print(f"LEN = {len(regions)}")
            continue
        regions = [region for region in regions if region != ""]

        for region in regions:
            lines = region.split("\n")
            try:
                start_lines.append(lines[0] + lines[1])
            except IndexError:
                start_lines.append(lines[0])

            try:
                end_lines.append(lines[-2] + lines[-1])
            except IndexError:
                end_lines.append(lines[-1])
            if len(region) < 10:
                print(repr(region))
        
        distance_matrix = create_distance_matrix(end_lines, start_lines, tokenizer, model)
        perm = sort_file_tsp(distance_matrix)
        regions_sorted = [regions[i] for i in perm]

        with open(os.path.join(FOLDER, "sorted", filename), "w") as f:
            for region in regions_sorted:
                if not region or region == "":
                    print("AAAAAAAAAAAAAAAAAAAAa")
                print(f"{region}\n", file=f)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
