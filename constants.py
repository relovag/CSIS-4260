import torch
import transformers

SETTINGS = {
    "image_enc": "tf_efficientnet_b0",
    "text_enc": "bert-base-uncased",
    "max_len": 32,
    "img_size": 256,
    "embed_size": 256,
    "train_bs": 32,
    "val_bs": 64,
    "T_max": 300,
    "eta": 1e-4,
    "eta_min": 1e-6,
    "eta_decay": 1e-6,
    "accum": 1,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}
SETTINGS["text_tknzr"] = transformers.AutoTokenizer.from_pretrained(
    SETTINGS["text_enc"])
