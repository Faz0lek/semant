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
    model = CzertNSP(len(tokenizer)) if args.czert else NSPModel()

    if args.model_path:
        print(f"Loading model from: {args.model_path} ...")
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    print(f"Model created. (n_params = {(n_params(model) / 1e6):.2f} M)")

    # Optimalizer, Loss
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.model_path:
        print(f"Loading optim from: {args.model_path} ...")
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Optim loaded.")

    criterion = nn.BCELoss()

    epoch_offset = checkpoint["epoch"] if args.model_path else 0

    # Training
    print(f"Starting training ...")
    for epoch in range(epoch_offset, args.epochs + epoch_offset):
        # Accumulators
        epoch_loss = 0.0
        steps_loss = 0.0
        val_loss = 0.0
        train_steps = 0
        val_steps = 0

        ground_truth = []
        predictions = []

        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            input_ids = inputs["input_ids"].squeeze(dim=1).to(device)
            token_type_ids = inputs["token_type_ids"].squeeze(dim=1).to(device)
            attention_mask = inputs["attention_mask"].squeeze(dim=1).to(device)
            labels = labels.unsqueeze(1).to(device=device, dtype=torch.float32)

            outputs = model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

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
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.clip)
            optim.step()

        # Validation
        model.eval()
        for batch_idx, (inputs, labels) in enumerate(val_loader):
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

            ground_truth.extend(labels.to(dtype=torch.int32).tolist())
            predictions.extend(outputs.detach().round().to(dtype=torch.int32).tolist())

            val_loss += loss.item()
            val_steps += 1

        # Print epoch loss
        print(f"Epoch {epoch+1} | Loss: {(epoch_loss / train_steps):.4f} | Val_Loss: {(val_loss / val_steps):.4f}")

        # Evaluate
        evaluate(ground_truth, predictions)

        # Save checkpoint
        checkpoint_path = os.path.join(args.save_path, f"checkpoint_{(epoch + 1):03d}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            }, checkpoint_path)

    print(f"Training finished.")

    # Test model
    predictions_test = []
    ground_truth_test = []
    test_loss = 0.0
    test_steps = 0
    
    model.eval()
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        input_ids = inputs["input_ids"].squeeze().to(device)
        token_type_ids = inputs["token_type_ids"].squeeze().to(device)
        attention_mask = inputs["attention_mask"].squeeze().to(device)
        labels = labels.to(device=device, dtype=torch.float32)

        outputs = model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ).squeeze()

        ground_truth_test.extend(labels.to(dtype=torch.int32).tolist())
        predictions_test.extend(outputs.detach().round().to(dtype=torch.int32).tolist())

    evaluate(ground_truth_test, predictions_test, full=False)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
