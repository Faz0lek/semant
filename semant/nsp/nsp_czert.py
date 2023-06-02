"""PyTorch NSP model using CZERT

Date -- 24.05.2023
Author -- Martin Kostelnik
"""

from torch import nn
from transformers import BertModel

from nsp_utils import CZERT_PATH


class CzertNSP(nn.Module):
    def __init__(self, embeddings_size: int, device):
        super(CzertNSP, self).__init__()

        self.device = device

        self.czert = BertModel.from_pretrained(CZERT_PATH)
        self.czert.resize_token_embeddings(embeddings_size)

        # self.classifier = nn.Sequential(
        #    nn.Linear(768, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 1),
        #    nn.Sigmoid(),
        #)
        self.classifier = nn.Sequential(
                nn.Linear(768, 1),
                nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.czert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        pooled_outputs = outputs[1]
        # pooled_outputs = self.dropout(pooled_outputs)

        p = self.classifier(pooled_outputs)

        return p
