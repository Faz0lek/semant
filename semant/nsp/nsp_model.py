"""PyTorch NSP model with custom architecture

Date -- 24.05.2023
Author -- Martin Kostelnik
"""

from torch import nn
from transformers import BertConfig, BertModel

from nsp_utils import CZERT_PATH
from configs import config_mapping


class NSPModel(nn.Module):
    def __init__(self, bert, device, name):
        super(NSPModel, self).__init__()

        self.name = name
        self.device = device
        self.bert = bert

        self.n_features = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
                nn.Linear(self.n_features, 1),
                nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        pooled_outputs = outputs[1]
        # pooled_outputs = self.dropout(pooled_outputs)

        p = self.classifier(pooled_outputs)

        return p


def build_model(
    czert: bool,
    vocab_size: int,
    device,
    out_features: int = None,
    ):
    assert (czert ^ bool(out_features))

    if czert:
        bert = BertModel.from_pretrained(CZERT_PATH)
        name = "CZERT"
    else:
        config = BertConfig.from_dict(config_mapping[f"bert_config_{out_features}"])
        bert = BertModel(config)
        name = f"Custom model with {out_features} features"

    bert.resize_token_embeddings(vocab_size)
    model = NSPModel(bert, device, name)

    return model
