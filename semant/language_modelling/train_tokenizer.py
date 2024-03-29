""" Train tokenizer on text corpus.

    Date -- 19.06.2023
    Author -- Martin Kostelnik
"""

import argparse
import os
import sys
from typing import List, Iterable

from transformers import BertTokenizerFast

from semant.language_modelling.utils import load_data, CZERT_PATH


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, type=str, help="Path to text corpus.")
    parser.add_argument("--save-path", required=True, type=str, help="Save path for tokenizer")
    parser.add_argument("--vocab-size", default=20000, type=int, help="Vocabulary size of the tokenizer.")

    args = parser.parse_args()
    return args


def get_training_corpus(data: List[str]) -> Iterable[List[str]]:
    return (data[i : i + 1000] for i in range(0, len(data), 1000))


def main(args):
    print("Loading text corpus ...")
    data = load_data(args.data, raw=True)
    print(f"Text corpus loaded, n_samples = {len(data)}")

    base_tokenizer = BertTokenizerFast.from_pretrained(CZERT_PATH)

    print("Starting training ...")
    tokenizer = base_tokenizer.train_new_from_iterator(get_training_corpus(data), args.vocab_size)
    print("Training finished.\nSaving ...")

    os.makedirs(args.save_path, exist_ok=True)
    tokenizer.save_pretrained(os.path.join(args.save_path, f"books-tokenizer-{args.vocab_size}"))


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
