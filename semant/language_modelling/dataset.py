"""PyTorch dataset for language modelling (BERT)

Date -- 24.05.2023
Author -- Martin Kostelnik
"""

import torch
from typing import List

from semant.language_modelling.tokenizer import LMTokenizer


class LMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: List[str],
        tokenizer: LMTokenizer,
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
    
    def __getitem__(self, index: int):
        sen1, sen2, nsp_label = self.data[index].strip().split("\t")
        
        sen1 = sen1.strip()
        sen2 = sen2.strip()

        tokenizer_output = self.tokenizer(sen1, sen2)
        tokenizer_output["sen1"] = sen1
        tokenizer_output["sen2"] = sen2
        
        mlm_labels = tokenizer_output["mlm_labels"]
        del tokenizer_output["mlm_labels"]

        return tokenizer_output, float(nsp_label), mlm_labels

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    from semant.language_modelling.tokenizer import build_tokenizer
    from semant.language_modelling.utils import load_data
    PATH = r"/home/martin/semant/data/books/books-dataset.tst"
    tokenizer = build_tokenizer(seq_len=80, fixed_sep=True)
    data_train = load_data(PATH)

    dataset = LMDataset(data_train, tokenizer)

    encoding, nsp_label, mlm_labels = dataset[1]

    ids = encoding["input_ids"].tolist()[0]
    attention_mask = encoding["attention_mask"].tolist()[0]
    token_type_ids = encoding["token_type_ids"].tolist()[0]
    ids_masked = encoding["input_ids_masked"].tolist()[0]
    mlm_labels = mlm_labels.tolist()[0]
    decoded_tokens = tokenizer.tokenizer.convert_ids_to_tokens(ids)
    decoded_mask = tokenizer.tokenizer.convert_ids_to_tokens(ids_masked)

    for i, (id, tok, at, typ, mask, dm, label) in enumerate(zip(ids, decoded_tokens, attention_mask, token_type_ids, ids_masked, decoded_mask, mlm_labels)):
        print(f"{i}\t{id}\t{tok}\t\t{at}\t{typ}\t{mask}\t{dm}\t{label}")
