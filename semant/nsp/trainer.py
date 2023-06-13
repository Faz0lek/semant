"""Trainer class for NSP models

Date -- 02.06.2023
Author -- Martin Kostelnik
"""

import argparse
import typing
from dataclasses import dataclass

import torch
from torch import nn

from nsp_utils import evaluate, accuracy


@dataclass
class TrainerSettings:
    lr: float
    clip: float
    view_step: int
    val_step: int
    epochs: int
    warmup_steps: int


@dataclass
class TrainerMonitor:
    train_loss: list
    val_loss: list
    train_acc: list
    val_acc: list


class Trainer:
    def __init__(self, model, tokenizer, settings: dict):
        self.settings = settings
        self.monitor = TrainerMonitor([], [], [], [])

        self.model = model
        self.tokenizer = tokenizer

        self.optim = torch.optim.Adam(self.model.parameters(), self.settings.lr, betas=(0.9, 0.98), eps=1e-9)
        self.criterion = nn.BCELoss()

    def train_step(self, batch, labels):
        loss, outputs = self.forward(batch, labels)
    
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.settings.clip)
        self.optim.step()

        predictions = outputs.detach().round().to(dtype=torch.int32)
        return loss.item(), predictions

    def test_step(self, batch, labels):
        with torch.no_grad():
            loss, outputs = self.forward(batch, labels)

        predictions = outputs.detach().round().to(dtype=torch.int32)
        return loss.item(), predictions

    def forward(self, batch, labels):
        device = self.model.device

        input_ids = batch["input_ids"].squeeze(dim=1).to(device)
        token_type_ids = batch["token_type_ids"].squeeze(dim=1).to(device)
        attention_mask = batch["attention_mask"].squeeze(dim=1).to(device)
        labels = labels.unsqueeze(1).to(device=device, dtype=torch.float32)

        outputs = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        loss = self.criterion(outputs, labels)

        return loss, outputs

    def lr_update(self, train_steps: int):
        d = self.model.n_features

        for group in self.optim.param_groups:
            group['lr'] = (d ** (-0.5)) * min(((train_steps+1) ** (-0.5)), (train_steps+1) * (self.warmup_steps ** (-1.5)))


    def train(self, train_loader, val_loader):
        train_gts = []
        train_preds = []
        steps_loss = 0.0
        train_steps = 0

        self.model.train()
        for epoch in range(self.settings.epochs):
            self.model.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if self.settings.warmup_steps:
                    lr_update(train_steps)

                loss, predictions = self.train_step(inputs, labels)

                steps_loss += loss
                train_steps += 1    
                train_gts.extend(labels.to(dtype=torch.int32).tolist())
                train_preds.extend(predictions.squeeze(dim=1).tolist())

                if not train_steps % self.settings.view_step:
                    display_loss = steps_loss / self.settings.view_step
                    print(f"Epoch {epoch+1} | Steps {train_steps} | Loss: {(display_loss):.4f}")

                    self.monitor.train_loss.append(display_loss)
                    self.monitor.train_acc.append(accuracy(train_gts, train_preds))

                    steps_loss = 0.0
                    train_gts = []
                    train_preds = []

                # Since we have a huge dataset, validate more often than once per epoch
                if not train_steps % self.settings.val_step:
                    self.validate(val_loader)
                    self.model.train()

            print(f"Epoch {epoch+1} finished.")

    def validate(self, loader, testing: bool=False):
        steps = 0
        total_loss = 0.0

        ground_truth = []
        all_predictions = []

        self.model.eval()
        for batch_idx, (inputs, labels) in enumerate(loader):
            loss, predictions = self.test_step(inputs, labels)

            total_loss += loss
            steps += 1

            ground_truth.extend(labels.to(dtype=torch.int32).tolist())
            all_predictions.extend(predictions.squeeze(dim=1).tolist())

        if not testing:
            display_loss = total_loss / steps
            print(f"Validation loss: {(total_loss / steps):.4f}")

            self.monitor.val_loss.append(display_loss)
            self.monitor.val_acc.append(accuracy(ground_truth, all_predictions))

        evaluate(ground_truth,
                 all_predictions,
                 self.monitor.train_loss,
                 self.monitor.val_loss,
                 self.monitor.train_acc,
                 self.monitor.val_acc,
                 self.settings.view_step,
                 self.settings.val_step,
                 full=testing)
