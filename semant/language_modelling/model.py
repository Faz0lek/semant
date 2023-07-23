"""PyTorch language model with BERT backend

Date -- 24.05.2023
Author -- Martin Kostelnik
"""

from typing import Callable, Optional

from torch import nn
from transformers import BertConfig, BertModel

from semant.language_modelling.utils import CZERT_PATH
from semant.language_modelling.configs import config_mapping


class CLSHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_layers: int = 1,
        hidden_size: int = 128,
        activation: Callable = nn.ReLU,
        output_size: int = 1,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.dropout_prob = dropout_prob

        self.layers = nn.ModuleList()

        if n_layers == 1:
            self.layers.append(nn.Linear(self.input_size, self.output_size))
        else:
            self.layers.append(nn.Linear(self.input_size, self.hidden_size))
            self.layers.append(self.activation)
            self.layers.append(nn.Dropout(self.dropout_prob))

            for layer_idx in range(n_layers - 2):
                self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
                self.layers.append(self.activation)
                self.layers.append(nn.Dropout(self.dropout_prob))

            self.layers.append(nn.Linear(self.hidden_size, self.output_size))

        self.layers.append(nn.Sigmoid() if output_size == 1 else nn.Softmax(dim=1))

    def forward(self, features):
        for layer in self.layers:
            features = layer(features)

        return features


class LanguageModelOutput():
    def __init__(
        self,
        mlm_output = None,
        nsp_output = None,
    ):
        self.mlm_output = mlm_output
        self.nsp_output = nsp_output


class LanguageModel(nn.Module):
    def __init__(self,
        bert: BertModel,
        device,
        name: str,
        nsp_head: CLSHead,
        mlm_head: Optional[CLSHead] = None,
        seq_len: int = 128,
        sep: bool = False,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.name = name
        self.device = device
        self.seq_len = seq_len
        self.sep = sep

        self.bert = bert
        self.mlm_head = mlm_head
        self.nsp_head = nsp_head

        self.n_features = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        # last_hidden_state
        # pooler_output
        # hidden_states
        # past_key_values
        # attentions
        # cross_attentions

        model_outputs = LanguageModelOutput()

        # if self.sep, take SEP token embedding, else take pooled CLS token embedding
        nsp_features = outputs.last_hidden_state[:, self.seq_len // 2, :] if self.sep else outputs[1]
        nsp_features = self.dropout(nsp_features)
        model_outputs.nsp_output = self.nsp_head(nsp_features)

        if self.mlm_head:
            mlm_features = outputs[0] # Contextual representations of all tokens (last hidden state)
            mlm_features = self.dropout(mlm_features)
            model_outputs.mlm_output = self.mlm_head(mlm_features)

        return model_outputs


def build_model(
    czert: bool,
    vocab_size: int,
    device,
    seq_len: int = 128,
    out_features: int = None,
    mlm_level: int = 0,
    sep: bool = False,
):
    if czert:
        bert = BertModel.from_pretrained(CZERT_PATH)
        name = "CZERT"
    else:
        config = BertConfig.from_dict(config_mapping[f"bert_config_{out_features}"])
        bert = BertModel(config)
        name = f"Custom model with {out_features} features"

    bert.resize_token_embeddings(vocab_size)
    n_features = bert.config.hidden_size

    nsp_head = CLSHead(n_features)
    mlm_head = CLSHead(n_features, output_size=vocab_size) if mlm_level == 2 else None

    model = LanguageModel(
        bert,
        device,
        name,
        nsp_head,
        mlm_head,
        seq_len,
        sep,
    )

    return model
