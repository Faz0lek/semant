"""Trainer class for NSP models

Date -- 02.06.2023
Author -- Martin Kostelnik
"""

import argparse
import typing
from dataclasses import dataclass

import torch
from torch import nn

from nsp_utils import evaluate


@dataclass
class TrainerSettings:
    lr: float
    clip: float
    view_step: int
    val_step: int
    epochs: int


class Trainer:
    def __init__(self, model, tokenizer, settings: dict):
        self.settings = settings

        self.model = model
        self.tokenizer = tokenizer

        self.optim = torch.optim.Adam(self.model.parameters(), self.settings.lr)
        self.criterion = nn.BCELoss()


    def train(self):
        pass

    def train_step(self, batch, labels):
        loss, _ = self.forward(batch, labels)
    
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.settings.clip)
        self.optim.step()

        return loss.item()

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

    def train(self, train_loader, val_loader):
        self.model.train()
        for epoch in range(self.settings.epochs):
            # Accumulators
            epoch_loss = 0.0
            steps_loss = 0.0
            train_steps = 0

            self.model.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                loss = self.train_step(inputs, labels)

                epoch_loss += loss
                steps_loss += loss
                train_steps += 1

                if not train_steps % self.settings.view_step:
                    print(f"Epoch {epoch+1} | Steps {train_steps} | Loss: {(steps_loss / self.settings.view_step):.4f}")
                    steps_loss = 0.0 

                # Since we have a huge dataset, validate more often than once per epoch
                if not train_steps % self.settings.val_step:
                    self.validate(val_loader)
                    self.model.train()

            # Validation at the end of an epoch
            self.validate(val_loader)
    
    def validate(self, loader):
        val_steps = 0
        val_loss = 0.0

        ground_truth = []
        all_predictions = []

        self.model.eval()
        for batch_idx, (inputs, labels) in enumerate(loader):
            loss, predictions = self.test_step(inputs, labels)

            val_loss += loss
            val_steps += 1

            ground_truth.extend(labels.to(dtype=torch.int32).tolist())
            all_predictions.extend(predictions.tolist())

        print(f"Validation loss: {(val_loss / val_steps):.4f}")
        evaluate(ground_truth, all_predictions, full=False)
