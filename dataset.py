import base64
import random
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class WikiDataset(Dataset):
    def __init__(self, data, max_len, tokenizer, transforms=None):
        self.data = data
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    # TODO getitem dunder
