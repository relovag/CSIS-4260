import base64
import random
from io import BytesIO

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from constants import SETTINGS


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
            'cap': cap,
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
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length"
        )

    @staticmethod
    def get_transforms(valid=False):
        mean, std, max_pix = [0.485, 0.456, 0.406], [
            0.229, 0.224, 0.225], 255.0
        transforms = {
            "train": A.Compose([
                A.Resize(SETTINGS['img_size'], SETTINGS['img_size']),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=max_pix,
                    p=1.0
                ),
                ToTensorV2()], p=1.0),

            "val": A.Compose([
                A.Resize(SETTINGS['img_size'], SETTINGS['img_size']),
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=max_pix,
                    p=1.0
                ),
                ToTensorV2()], p=1.)
        }
        return transforms["val"] if valid else transforms["train"]
