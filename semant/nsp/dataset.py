"""PyTorch dataset for NSP

Date -- 24.05.2023
Author -- Martin Kostelnik
"""

import torch


class NSPDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        super(NSPDataset, self).__init__()

        self.data = data
        self.tokenizer = tokenizer

        self.CLS_TOKEN = 2
        self.SEP_TOKEN = 3
        self.PAD_TOKEN = 0
    
    def __getitem__(self, index: int, fixed_sep: bool = False):
        sen1, sen2, label = self.data[index].strip().split("\t")
        
        sen1 = sen1.strip()
        sen2 = sen2.strip()

        if fixed_sep:
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
            sen1_encoded = sen1_encoded[:-15]

        if len(sen2_encoded) > 15:
            sen2_encoded = sen2_encoded[:-15]

        padding_left_len = 15 - len(sen1_encoded)
        padding_right_len = 15 - len(sen2_encoded)
        padding_left = [self.PAD_TOKEN] * padding_left_len
        padding_right = [self.PAD_TOKEN] * padding_right_len
        result = padding_left + [self.CLS_TOKEN] + sen1_encoded + [self.SEP_TOKEN] + sen1_encoded + [self.SEP_TOKEN] + padding_right
        
        return result


if __name__ == "__main__":
    from nsp_utils import build_tokenizer, load_data
    PATH = r"/home/martin/semant/data/books/books-dataset.tst"
    tokenizer = build_tokenizer()
    data_train = load_data(PATH)
    dataset = NSPDataset(data_train, tokenizer)

    ids, label = dataset[1]

    decoded_tokens = tokenizer.convert_ids_to_tokens(ids)
    print(ids)
    print(decoded_tokens, label)
