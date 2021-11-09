import base64
import random
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class WikiDataset(Dataset):
    def __init__(self, data, max_len, tokenizer, transforms):
        self.data = data
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.convert_image(index)
        cap = self.get_caption(index)
        ip = self.tokenize(cap)
        target = self.data[index]["target"]
        ids = ip["input_ids"]
        mask = ip["attention_mask"]
        img = self.transforms(image=img)["image"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'image': img,
            'target': torch.tensor(target, dtype=torch.long)
        }

    def convert_image(self, index):
        bin_image = base64.b64decode(self.data[index]["b64_bytes"])
        return np.asarray(Image.open(BytesIO(bin_image)).convert("RGB"))

    def get_caption(self, index):
        return random.choice(self.data[index]["caption_title_and_reference_description"])

    def tokenize(self, cap):
        return self.tokenizer.encode_plus(
            cap,
            add_special_token=True,
            max_length=self.max_len,
            padding="max_length"
        )
