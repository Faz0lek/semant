"""PyTorch NSP model with custom architecture

Date -- 24.05.2023
Author -- Martin Kostelnik
"""

from torch import nn


class NSPModel(nn.Module):
    def __init__(self):
        super(NSPModel, self).__init__()

        self.feature_extractor = nn.Sequential()

        self.classifier = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, inputs):
        features = self.feature_extractor(inputs)
        logits = self.classifier(features)
        return logits
