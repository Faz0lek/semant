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
from transformers import BertTokenizerFast

from dataset import NSPDataset
from nsp_utils import CZERT_PATH, load_data
from nsp_czert import CzertNSP
from nsp_model import NSPModel


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="Path to dataset.")
    parser.add_argument("--czert", action="store_true", help="Train baseline CZERT instead of our model.")

    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--view-step", type=int, default=1000, help="How often to print train loss.")

    parser.add_argument("--save-path", default=".", type=str, help="Model checkpoints will be saved here.")
    parser.add_argument("--model-path", default=None, type=str, help="Load model from saved checkpoint.")

    args = parser.parse_args()
    return args


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(CZERT_PATH)
    print(f"Tokenizer created.")

    # Data
    print(f"Loading data ...")
    data = load_data(args.data)
    dataset = NSPDataset(data, tokenizer)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Data loaded. n_samples = {len(dataset)}")

    # Model
    print(f"Creating {'CZERT' if args.czert else 'NSPModel'} model ...")
    model = CzertNSP() if args.czert else NSPModel()

    if args.model_path:
        print(f"Loading model from: {args.model_path} ...")
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    print(f"Model created.")

    # Hyperparameters
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.model_path:
        print(f"Loading optim from: {args.model_path} ...")
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Optim loaded.")

    criterion = nn.BCELoss()

    epoch_offset = checkpoint["epoch"] if args.model_path else 0

    # Training
    print(f"Starting training ...")
    model.train()
    for epoch in range(epoch_offset, args.epochs + epoch_offset):
        # Accumulators
        epoch_loss = 0.0
        steps_loss = 0.0
        train_steps = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            input_ids = inputs["input_ids"].squeeze().to(device)
            token_type_ids = inputs["token_type_ids"].squeeze().to(device)
            attention_mask = inputs["attention_mask"].squeeze().to(device)
            labels = labels.to(device=device, dtype=torch.float32)

            outputs = model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            ).squeeze()

            loss = criterion(outputs, labels)

            # Update accumulators
            epoch_loss += loss.item()
            steps_loss += loss.item()
            train_steps += 1

            # Print steps loss
            if not train_steps % args.view_step:
                print(f"Epoch {epoch+1} | Steps {train_steps} | Loss: {(steps_loss / args.view_step):.4f}")
                steps_loss = 0.0

            # Step
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Print epoch loss
        print(f"Epoch {epoch+1} | Loss: {(epoch_loss / train_steps):.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(args.save_path, f"checkpoint_{(epoch + 1):03d}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            }, checkpoint_path)

    print(f"Training finished.")

    # Model evaluation here?


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
