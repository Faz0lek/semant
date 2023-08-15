"""Test model on Babicka book.

    Date -- 16.07.2023
    Author -- Martin Kostelnik
"""

import argparse
import sys
import logging
from itertools import pairwise
from tqdm import tqdm

import torch

from semant.language_modelling.tokenizer import build_tokenizer
from semant.language_modelling.model import build_model


def parse_arguments():
    logging.info(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--file", required=True, type=str, help="Path to file to be sorted.")
    parser.add_argument("--model-path", required=True, type=str, help="Path to trained model.")
    parser.add_argument("--tokenizer-path", required=True, type=str, help="Path to trained tokenizer.")

    args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(level=logging.DEBUG, force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on: {device}")

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
    with open(args.file, "r") as f:
        babicka = f.read().replace("\n\n", "\n")
    logging.info("Input file loaded.")

    lines = babicka.split("\n")

    hits = 0
    for sen1, sen2 in tqdm(pairwise(lines)):
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

        if score < 0.5:
            hits += 1

    acc = (hits / (len(lines) - 1)) * 100.0
    print(f"{acc:.4f}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
