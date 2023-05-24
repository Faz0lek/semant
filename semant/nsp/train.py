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
from nsp_utils import CZERT_PATH, MODELS_PATH, load_data
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
    model = model.to(device)
    print(f"Model created.")

    # Hyperparameters
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Training
    print(f"Starting training ...")
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        if not i:
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

            optim.zero_grad()
            loss.backward()
            optim.step()

    print(f"Training finished.")

    # Save model
    # save_path = os.path.join(MODELS_PATH, "czert1")#str(datetime.datetime.now()))
    # os.makedirs(save_path, exist_ok=True)
    # torch.save(model.state_dict, save_path)

    # Testing


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
