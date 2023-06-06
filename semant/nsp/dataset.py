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
    
    def __getitem__(self, index):
        sen1, sen2, label = self.data[index].strip().split("\t")
        
        sen1 = sen1.strip()
        sen2 = sen2.strip()

        tokenizer_output = self.tokenizer(
            sen1,
            sen2,
            max_length=80,
            padding="max_length",
            return_tensors="pt")

        return tokenizer_output, float(label)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    from nsp_utils import build_tokenizer, load_data
    PATH = r"/home/martin/semant/data/books/books-dataset.tst"
    tokenizer = build_tokenizer()
    data_train = load_data(PATH)
    dataset = NSPDataset(data_train, tokenizer)

    outputs, label = dataset[1]
    ids = outputs["input_ids"].tolist()[0]

    decoded_tokens = tokenizer.convert_ids_to_tokens(ids)
    print(decoded_tokens, label)
