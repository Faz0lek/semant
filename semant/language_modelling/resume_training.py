"""Script to continue training already pre-trained
BERT-like language model

Date -- 20.07.2023
Author -- Martin Kostelnik
"""

import argparse
import sys
from time import perf_counter

import torch

from semant.language_modelling.tokenizer import build_tokenizer
from semant.language_modelling.train import prepare_loaders
from semant.language_modelling.model import build_model
from semant.language_modelling.utils import n_params
from semant.language_modelling.trainer import Trainer, TrainerSettings


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train", required=True, help="Path to train dataset.")
    parser.add_argument("--test", required=True, help="Path to test dataset.")
    parser.add_argument("--split", type=float, default=0.8, help="Train - validation split.")

    # MLM
    parser.add_argument("--mlm-level", type=int, default=2, choices=[0, 1, 2], help="0 -- no MLM ; 1 -- Masking only; 2 -- Masking + MLM loss.")
    parser.add_argument("--masking-prob", type=float, default=0.15, help="Masking probability for MLM.")
    # Trainer settings
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--steps", type=int, default=-1, help="Limit number of steps for finer control of training time.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--clip", type=float, default=3.0, help="Gradient clipping.")
    parser.add_argument("--view-step", type=int, default=100, help="How often to print train loss.")
    parser.add_argument("--val-step", type=int, default=10000, help="How often to validate model.")

    # Save/load paths
    parser.add_argument("--save-path", default=".", type=str, help="Model checkpoints will be saved here.")
    parser.add_argument("--model-path", required=True, type=str, help="Load model from saved checkpoint.")
    parser.add_argument("--tokenizer-path", default=None, type=str, help="Path to pretrained tokenizer.")

    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    print(f"Loading checkpoint from {args.model_path} ...")
    checkpoint = torch.load(args.model_path)
    print("Checkpoint loaded.")

    print("Creating tokenizer ...")
    tokenizer = build_tokenizer(
        args.tokenizer_path,
        seq_len=checkpoint["seq_len"],
        fixed_sep=checkpoint["fixed_sep"],
        masking_prob=args.masking_prob,
    )
    print("Tokenizer created.")

    print("Creating loaders ...")
    train_loader, val_loader, test_loader = prepare_loaders(
        args.train,
        args.test,
        tokenizer,
        args.batch_size,
        args.split
    )
    print("Loaders created ...")

    print("Creating model ...")
    model = build_model(
        czert=checkpoint["czert"],
        vocab_size=len(tokenizer),
        device=device,
        seq_len=checkpoint["seq_len"],
        out_features=checkpoint["features"],
        mlm_level=args.mlm_level,
        sep=checkpoint["sep"],
    )
    model.bert.load_state_dict(checkpoint["bert_state_dict"])
    model.nsp_head.load_state_dict(checkpoint["nsp_head_state_dict"])
    if checkpoint["mlm_head_state_dict"] and args.mlm_level == 2:
        model.mlm_head.load_state_dict(checkpoint["mlm_head_state_dict"])

    model.to(device)
    print(f"{model.name} created. (n_params = {(n_params(model) / 1e6):.2f} M)")

    print("Creating Trainer instance ...")
    trainer_settings = TrainerSettings(
        mlm_level=args.mlm_level,
        lr=args.lr,
        clip=args.clip,
        view_step=args.view_step,
        val_step=args.val_step,
        epochs=args.epochs,
        warmup_steps=0,
        save_path=args.save_path,
        fixed_sep=checkpoint["fixed_sep"],
        steps_limit=args.steps + checkpoint["steps"],
    )

    trainer = Trainer(model, tokenizer, trainer_settings)
    print("Trainer created.")

    start = perf_counter()
    print("Starting training ...")
    try:
        trainer.train(train_loader, val_loader, checkpoint["steps"], checkpoint["epochs"])
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
    args = parse_arguments()
    main(args)
