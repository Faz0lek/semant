"""PyTorch dataset for NSP

Date -- 24.05.2023
Author -- Martin Kostelnik
"""

import torch


class NSPDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, sep_pos: int = 0):
        super(NSPDataset, self).__init__()

        self.sep_pos = sep_pos
        self.data = data
        self.tokenizer = tokenizer

        self.CLS_TOKEN = 2
        self.SEP_TOKEN = 3
        self.PAD_TOKEN = 0
    
    def __getitem__(self, index: int):
        sen1, sen2, label = self.data[index].strip().split("\t")
        
        sen1 = sen1.strip()
        sen2 = sen2.strip()

        if self.sep_pos:
            return self.get_fixed_sep_sequence(sen1, sen2), float(label)

        tokenizer_output = self.tokenizer(
            sen1,
            sen2,
            max_length=150,
            padding="max_length",
            return_tensors="pt")

        return tokenizer_output, float(label)

    def __len__(self):
        return len(self.data)

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

        if len(sen1_encoded) > 15:
            sen1_encoded = sen1_encoded[-15:]

        if len(sen2_encoded) > 15:
            sen2_encoded = sen2_encoded[-15:]

        if len(sen1_encoded) > 15 or len(sen2_encoded) > 15:
            print("AAAAAAAAAAAAa")

        padding_left_len = 15 - len(sen1_encoded)
        padding_right_len = 15 - len(sen2_encoded)
        padding_left = [self.PAD_TOKEN] * padding_left_len
        padding_right = [self.PAD_TOKEN] * padding_right_len
        
        left = padding_left + [self.CLS_TOKEN] + sen1_encoded + [self.SEP_TOKEN]
        right = sen2_encoded + [self.SEP_TOKEN] + padding_right
        result["input_ids"] =  left + right
        result["attention_mask"] = [int(bool(id)) for id in result["input_ids"]]
        result["token_type_ids"] = [0] * len(left) + [1] * (len(right) - padding_right_len) + [0] * padding_right_len

        if len(result["input_ids"]) != len(result["token_type_ids"]):
            print(len(right), len(right) - padding_right_len, padding_right_len)

        for key, val in result.items():
            result[key] = torch.tensor([val])

        return result


if __name__ == "__main__":
    from nsp_utils import build_tokenizer, load_data
    PATH = r"/home/martin/semant/data/books/books-dataset.tst"
    tokenizer = build_tokenizer()
    data_train = load_data(PATH)

    dataset_fixed = NSPDataset(data_train, tokenizer, True)
    dataset = NSPDataset(data_train, tokenizer, False)

    fixed_encoding, label = dataset_fixed[1]
    encoding, _ = dataset[1]

    idss = fixed_encoding["input_ids"].tolist()[0]
    attention_mask = fixed_encoding["attention_mask"].tolist()[0]
    token_type_ids = fixed_encoding["token_type_ids"].tolist()[0]
    
    decoded_tokens = tokenizer.convert_ids_to_tokens(idss)

    for id, tok, at, typ in zip(idss, decoded_tokens, attention_mask, token_type_ids):
        print(f"{id}\t{tok}\t\t{at}\t{typ}")
