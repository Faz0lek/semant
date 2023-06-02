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
            max_length=150,
            padding="max_length",
            return_tensors="pt")

        return tokenizer_output, float(label)

    def __len__(self):
        return len(self.data)
