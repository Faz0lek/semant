"""Bert-like language model training.

Date -- 24.05.2023
Author -- Martin Kostelnik
"""

import argparse
import sys
from typing import Tuple
import os
from time import perf_counter

import torch
from torch import nn

from dataset import LMDataset
from utils import build_tokenizer, load_data, n_params, evaluate
from model import build_model
from trainer import Trainer, TrainerSettings

from transformers import BertTokenizerFast

from safe_gpu import safe_gpu


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train", required=True, help="Path to train dataset.")
    parser.add_argument("--test", required=True, help="Path to test dataset.")
    parser.add_argument("--split", type=float, default=0.8, help="Train - validation split.")

    # Backend
    group_model = parser.add_mutually_exclusive_group(required=True)
    group_model.add_argument("--czert", action="store_true", help="Train baseline CZERT instead of our model.")
    group_model.add_argument("--features", type=int, default=0, choices=[0, 72, 132, 264, 516], help="Number of features of BERT model.")
    
    # MLM
    parser.add_argument("--mlm-level", type=int, default=0, choices=[0, 1, 2], help="0 -- no MLM ; 1 -- Masking only; 2 -- Masking + MLM loss.")

    # Trainer settings
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--steps", type=int, default=-1, help="Limit number of steps for finer control of training time.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping.")
    parser.add_argument("--view-step", type=int, default=1000, help="How often to print train loss.")
    parser.add_argument("--val-step", type=int, default=5000, help="How often to validate model.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="How many warmup steps (linear warmup).")
    parser.add_argument("--seq-len", type=int, default=128, help="Maximum length of input sequence.")
    parser.add_argument("--fixed-sep", action="store_true", help="Use sequences with fixed SEP token.")
    parser.add_argument("--sep", action="store_true", help="Use SEP token contextual embedding insead of CLS for NSP. SEP token must be fixed (--fixed_sep)")

    # Save/load paths
    parser.add_argument("--save-path", default=".", type=str, help="Model checkpoints will be saved here.")
    parser.add_argument("--model-path", default=None, type=str, help="Load model from saved checkpoint.")
    parser.add_argument("--tokenizer-path", default=None, type=str, help="Path to pretrained tokenizer.")

    args = parser.parse_args()
    return args


def prepare_loaders(
    train_path: str,
    test_path: str,
    tokenizer: BertTokenizerFast,
    batch_size: int,
    ratio: float,
    seq_len: int,
    fixed: bool,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    start = perf_counter()
    print(f"Loading train data from {train_path} ...")
    data_train = load_data(train_path)
    dataset = LMDataset(data_train, tokenizer, seq_len=seq_len, fixed=fixed)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [ratio, 1 - ratio])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_loaders=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_loaders=1)
    end = perf_counter()
    t = end - start
    print(f"Train data loaded. n_samples = {len(dataset)}\ttrain = {len(train_dataset)}\tval = {len(val_dataset)}\ttook {(t / 60):.1f} m")

    start = perf_counter()
    print(f"Loading test data from {test_path} ...")
    data_test = load_data(test_path)
    test_dataset = LMDataset(data_test, tokenizer, seq_len=seq_len, fixed=fixed)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    end = perf_counter()
    t = end - start
    print(f"Test data loaded. n_samples = {len(test_dataset)}\t took {(t / 60):.1f} m")

    return train_loader, val_loader, test_loader


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Tokenizer
    tokenizer = build_tokenizer(args.tokenizer_path)
    print(f"Tokenizer created.")

    # Data
    train_loader, val_loader, test_loader = prepare_loaders(
        args.train,
        args.test,
        tokenizer,
        args.batch_size,
        args.split,
        args.seq_len,
        args.fixed_sep,
    )

    # Model
    print(f"Creating model ...")
    model = build_model(
        args.czert,
        len(tokenizer),
        device,
        args.seq_len,
        args.features,
        args.mlm_level,
        args.sep,
    )

    if args.model_path:
        print(f"Loading model from: {args.model_path} ...")
        model.load_state_dict(torch.load(args.model_path))

    model = model.to(device)
    print(f"{model.name} created. (n_params = {(n_params(model) / 1e6):.2f} M)")

    # Trainer settings
    print("Creating Trainer instance ...")
    trainer_settings = TrainerSettings(
        mlm_level=args.mlm_level,
        lr=args.lr,
        clip=args.clip,
        view_step=args.view_step,
        val_step=args.val_step,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        save_path=args.save_path,
        fixed_sep=args.fixed_sep,
        steps_limit=args.steps,
    )

    trainer = Trainer(model, tokenizer, trainer_settings)
    print("Trainer created.")

    start = perf_counter()
    print("Starting training ...")
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("Training stopped by user.")
    finally:
        # Testing
        end = perf_counter()
        t = end - start
        print(f"Training finished. Took {(t / 60):.1f} m")
        print("Testing ...")
        trainer.validate(test_loader, True)


if __name__ == "__main__":
    safe_gpu.claim_gpus()
    args = parse_arguments()
    main(args)
