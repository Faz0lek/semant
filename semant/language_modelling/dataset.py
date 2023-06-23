"""PyTorch dataset for language modelling (BERT)

Date -- 24.05.2023
Author -- Martin Kostelnik
"""

import torch


class LMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        seq_len: int = 128,
        fixed: bool = False,
        maskin_prob: float = 0.15,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.fixed = fixed
        self.sep_pos = seq_len // 2 if self.fixed else None
        self.masking_prob = maskin_prob
        self.data = data
        self.tokenizer = tokenizer

        # self.CLS_TOKEN = 2
        # self.SEP_TOKEN = 3
        # self.PAD_TOKEN = 0
        # self.MASK_TOKEN = 4
        self.CLS_TOKEN = tokenizer.cls_token_id
        self.SEP_TOKEN = tokenizer.sep_token_id
        self.PAD_TOKEN = tokenizer.pad_token_id
        self.MASK_TOKEN = tokenizer.mask_token_id
    
    def __getitem__(self, index: int):
        sen1, sen2, nsp_label = self.data[index].strip().split("\t")
        
        sen1 = sen1.strip()
        sen2 = sen2.strip()

        if self.fixed: # Sequence length is fixed
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

        return tokenizer_output, float(nsp_label), mlm_labels

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

    def get_masked_input_ids(self, input_ids):
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


if __name__ == "__main__":
    from utils import build_tokenizer, load_data
    PATH = r"/home/martin/semant/data/books/books-dataset.tst"
    tokenizer = build_tokenizer()
    data_train = load_data(PATH)

    dataset = LMDataset(data_train, tokenizer, seq_len=80, fixed=True)

    encoding, nsp_label, mlm_labels = dataset[1]

    ids = encoding["input_ids"].tolist()[0]
    attention_mask = encoding["attention_mask"].tolist()[0]
    token_type_ids = encoding["token_type_ids"].tolist()[0]
    ids_masked = encoding["input_ids_masked"].tolist()[0]
    mlm_labels = mlm_labels.tolist()[0]
    decoded_tokens = tokenizer.convert_ids_to_tokens(ids)
    decoded_mask = tokenizer.convert_ids_to_tokens(ids_masked)

    for i, (id, tok, at, typ, mask, dm, label) in enumerate(zip(ids, decoded_tokens, attention_mask, token_type_ids, ids_masked, decoded_mask, mlm_labels)):
        print(f"{i}\t{id}\t{tok}\t\t{at}\t{typ}\t{mask}\t{dm}\t{label}")
