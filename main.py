import pickle

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from constants import SETTINGS
from dataset import WikiDataset
from train import ModelTrainer
from model import ImageEncoder, TextEncoder


def run():
    with open('data/wiki_data.pkl', 'rb') as f:
        data = pickle.load(f)

    train, val = train_test_split(data, random_state=42)

    train_data = WikiDataset(train, SETTINGS['max_len'], SETTINGS['text_tknzr'],
                             transforms=None)
    train_loader = DataLoader(
        train_data, batch_size=SETTINGS['train_bs'], shuffle=True)

    val_data = WikiDataset(val, SETTINGS['max_len'], SETTINGS['text_tknzr'],
                           transforms=None)
    val_loader = DataLoader(
        val_data, batch_size=SETTINGS['val_bs'], shuffle=True)

    img_enc = ImageEncoder(SETTINGS["image_enc"], SETTINGS["embed_size"])
    img_enc.to(SETTINGS["device"])

    txt_enc = TextEncoder(SETTINGS["text_enc"], SETTINGS["embed_size"])
    txt_enc.to(SETTINGS["device"])

    mod_trainer = ModelTrainer(
        img_enc, txt_enc, train_loader, val_loader, SETTINGS["device"])
    history = mod_trainer.train_multiple_eps(2)

    with open('data/history.pkl', 'wb') as f:
        pickle.dump(history, f)


if __name__ == '__main__':
    run()
