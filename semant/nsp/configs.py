"""Configs for several small BERT models.

    Date -- 06.06.2023
    Author -- Martin Kostelnik
"""

from transformers import BertConfig, BertModel
from nsp_utils import n_params, model_size

"""Description

    https://huggingface.co/docs/transformers/v4.29.1/en/model_doc/bert#transformers.BertConfig

    attention_probs_dropout_prob : The dropout ratio for the attention probabilities
    classifier_dropout : null -- The dropout ratio for the classification head
    hidden_act : The non-linear activation function (function or string) in the encoder and pooler. If string, "gelu", "relu", "silu" and "gelu_new" are supported.
    hidden_dropout_prob : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
    hidden_size : Dimensionality of the encoder layers and the pooler layer
    initializer_range : The standard deviation of the truncated_normal_initializer for initializing all weight matrices
    intermediate_size : Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder
    layer_norm_eps : The epsilon used by the layer normalization layers
    max_position_embeddings : The maximum sequence length that this model might ever be used with
    num_attention_heads : Number of attention heads for each attention layer in the Transformer encoder
    num_hidden_layers : Number of hidden layers in the Transformer encoder
    position_embedding_type : Type of position embedding. Choose one of "absolute", "relative_key", "relative_key_query". For positional embeddings use "absolute". 
    type_vocab_size : The vocabulary size of the token_type_ids passed when calling BertModel
    vocab_size : Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the inputs_ids passed when calling BertModel
"""

bert_config_64 = {
    "attention_probs_dropout_prob": 0.1,
    "classifier_dropout": None,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 64,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 8,
    "num_hidden_layers": 8,
    "position_embedding_type": "absolute",
    "vocab_size": 30522,
}


bert_config_128 = {
    "attention_probs_dropout_prob": 0.1,
    "classifier_dropout": None,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 128,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 8,
    "num_hidden_layers": 8,
    "position_embedding_type": "absolute",
    "vocab_size": 30522,
}


bert_config_256 = {
    "attention_probs_dropout_prob": 0.1,
    "classifier_dropout": None,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 8,
    "num_hidden_layers": 8,
    "position_embedding_type": "absolute",
    "vocab_size": 30522,
}


bert_config_512 = {
    "attention_probs_dropout_prob": 0.1,
    "classifier_dropout": None,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 512,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 8,
    "num_hidden_layers": 8,
    "position_embedding_type": "absolute",
    "vocab_size": 30522,
}


if __name__ == "__main__":
    config = bert_config_512
    config = BertConfig.from_dict(config)
    # print(config)

    model = BertModel(config)
    # print(model)

    print(f"n_params = {n_params(model) / 1e6} M")
    print(f"size = {model_size(model) / 1024**2} MB")
