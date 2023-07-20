"""PyTorch dataset for language modelling (BERT)

Date -- 24.05.2023
Author -- Martin Kostelnik
"""

from typing import Dict
from dataclasses import dataclass

import torch
from torch import FloatTensor
from transformers import BertTokenizerFast

from semant.language_modelling.utils import CZERT_PATH, JOKER


class LMTokenizer:
    def __init__(
        self,
        tokenizer: BertTokenizerFast,
        seq_len: int=128,
        fixed_sep: bool=False,
        masking_prob: float=0.15,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.fixed_sep = fixed_sep
        self.masking_prob = masking_prob
        self.sep_pos = seq_len // 2 if self.fixed_sep else None

        self.CLS_TOKEN = self.tokenizer.cls_token_id
        self.SEP_TOKEN = self.tokenizer.sep_token_id
        self.PAD_TOKEN = self.tokenizer.pad_token_id
        self.MASK_TOKEN = self.tokenizer.mask_token_id

    def __call__(
        self,
        sen1: str,
        sen2: str,
    ) -> Dict[str, FloatTensor]:
        sen1 = sen1.strip()
        sen2 = sen2.strip()

        if self.fixed_sep:
            tokenizer_output = self.get_fixed_sep_sequence(sen1, sen2)
        else:
            tokenizer_output = self.tokenizer(
                sen1,
                sen2,
                max_length=self.seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

        input_ids_masked, mlm_labels = self.get_masked_input_ids(tokenizer_output["input_ids"])
        tokenizer_output["input_ids_masked"] = input_ids_masked
        tokenizer_output["mlm_labels"] = mlm_labels

        return tokenizer_output

    def __len__(self):
        return len(self.tokenizer)

    def get_fixed_sep_sequence(self, sen1: str, sen2: str):
        result = {}

        sen1_encoded = self.tokenizer(
            sen1,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"][1:-1]

        sen2_encoded = self.tokenizer(
            sen2,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"][1:-1]

        # If sequence length is even, the second sentence will be one token longer
        left_len = int((self.seq_len - 3) / 2 + 0.5)
        right_len = (self.seq_len - 3) - left_len

        if len(sen1_encoded) > left_len:
            sen1_encoded = sen1_encoded[-left_len:]

        if len(sen2_encoded) > right_len:
            sen2_encoded = sen2_encoded[-right_len:]

        padding_left_len = left_len - len(sen1_encoded)
        padding_right_len = right_len - len(sen2_encoded)
        padding_left = [self.PAD_TOKEN] * padding_left_len
        padding_right = [self.PAD_TOKEN] * padding_right_len
        
        left = [self.CLS_TOKEN] + padding_left + sen1_encoded + [self.SEP_TOKEN]
        right = sen2_encoded + [self.SEP_TOKEN] + padding_right
        result["input_ids"] =  left + right
        result["attention_mask"] = [int(bool(id)) for id in result["input_ids"]]
        result["token_type_ids"] = [0] * len(left) + [1] * (len(right) - padding_right_len) + [0] * padding_right_len

        for key, val in result.items():
            result[key] = torch.tensor([val])

        return result
    
    def get_masked_input_ids(self, input_ids: torch.FloatTensor):
        shape = input_ids.size()
        labels = torch.ones(*shape, dtype=torch.int64) * -100
        masked_ids = input_ids.clone()

        for i, id in enumerate(input_ids.squeeze()):
            if id in [self.CLS_TOKEN, self.PAD_TOKEN, self.SEP_TOKEN]:
                continue

            masking_p = torch.rand(1).item()

            if masking_p < self.masking_prob:
                p = torch.rand(1).item()
                labels[0, i] = id

                if p < 0.8: # Mask token
                    masked_ids[0, i] = self.MASK_TOKEN
                elif p < 0.9: # Random token
                    random_token = torch.randint(low=5, high=len(self.tokenizer), size=(1,)).item()
                    masked_ids[0, i] = random_token
                else: # Keep token
                    labels[0, i] = id

        return masked_ids, labels


def build_tokenizer(
        path: str=None,
        seq_len: int=80,
        fixed_sep: bool=True,
        masking_prob: float=0.15,
    ) -> LMTokenizer:
    if path is None: # Create CZERT tokenizer
        base_tokenizer = BertTokenizerFast.from_pretrained(CZERT_PATH)
        base_tokenizer.add_special_tokens({"additional_special_tokens": [JOKER]})
    else: # Load tokenizer from path
        base_tokenizer = BertTokenizerFast.from_pretrained(path)

    tokenizer = LMTokenizer(
        base_tokenizer,
        seq_len,
        fixed_sep,
        masking_prob,
    )
    
    return tokenizer
