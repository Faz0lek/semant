"""NSP model training.

Date -- 24.05.2023
Author -- Martin Kostelnik
"""

import argparse
import sys
import typing
import os
import datetime

import torch
from torch import nn

from dataset import NSPDataset
from nsp_utils import build_tokenizer, load_data, n_params, evaluate
from nsp_czert import CzertNSP
from nsp_model import NSPModel
from trainer import Trainer, TrainerSettings


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", required=True, help="Path to train dataset.")
    parser.add_argument("--test", required=True, help="Path to test dataset.")
    parser.add_argument("--czert", action="store_true", help="Train baseline CZERT instead of our model.")

    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--view-step", type=int, default=1000, help="How often to print train loss.")
    parser.add_argument("--val-step", type=int, default=5000, help="How often to validate model.")
    parser.add_argument("--split", type=float, default=0.8, help="Train - validation split.")

    parser.add_argument("--save-path", default=".", type=str, help="Model checkpoints will be saved here.")
    parser.add_argument("--model-path", default=None, type=str, help="Load model from saved checkpoint.")

    args = parser.parse_args()
    return args


def prepare_loaders(train_path: str, test_path: str, tokenizer, batch_size: int, ratio: float) -> tuple:
    print(f"Loading train data from {train_path} ...")
    data_train = load_data(train_path)
    dataset = NSPDataset(data_train, tokenizer)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [ratio, 1 - ratio])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    print(f"Train data loaded. n_samples = {len(dataset)}\ttrain = {len(train_dataset)}\tval = {len(val_dataset)}")

    print(f"Loading test data from {test_path} ...")
    data_test = load_data(test_path)
    test_dataset = NSPDataset(data_test, tokenizer)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    print(f"Test data loaded. n_samples = {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Tokenizer
    tokenizer = build_tokenizer()
    print(f"Tokenizer created.")

    # Data
    train_loader, val_loader, test_loader = prepare_loaders(args.train, args.test, tokenizer, args.batch_size, args.split)

    # Model
    print(f"Creating {'CZERT' if args.czert else 'NSPModel'} model ...")
    model = CzertNSP(len(tokenizer), device) if args.czert else NSPModel()

    if args.model_path:
        print(f"Loading model from: {args.model_path} ...")
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    print(f"Model created. (n_params = {(n_params(model) / 1e6):.2f} M)")

    # Trainer settings
    print("Creating Trainer instance ...")
    trainer_settings = TrainerSettings(
        lr=args.lr,
        clip=args.clip,
        view_step=args.view_step,
        val_step=args.val_step,
        epochs=args.epochs,
    )

    trainer = Trainer(model, tokenizer, trainer_settings)
    print("Trainer created.")

    print("Starting training ...")
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("Training stopped by user. Validating:")
        trainer.validate(val_loader)
    print("Training finished.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
