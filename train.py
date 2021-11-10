import copy
import gc
from collections import defaultdict
from time import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from timm.optim.lamb import Lamb
from torch.optim.lr_scheduler import CosineAnnealingLR

from constants import SETTINGS


class ModelTrainer:
    def __init__(self, img_enc, txt_enc, train_loader, val_loader, device, img_optim=None,
                 txt_optim=None, img_scheduler=None, txt_scheduler=None, criterion=None):
        self.img_enc = img_enc
        self.txt_enc = txt_enc
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.img_optim = img_optim if img_optim else self.init_optim('img')
        self.txt_optim = txt_optim if txt_optim else self.init_optim('txt')
        self.img_scheduler = img_scheduler if img_scheduler else self.init_scheduler(
            'img')
        self.txt_scheduler = txt_scheduler if txt_scheduler else self.init_scheduler(
            'txt')
        self.criterion = criterion if criterion else nn.CosineEmbeddingLoss()

    def train_one_ep(self, epoch, valid=False):
        self.img_enc.eval() if valid else self.img_enc.train()
        self.txt_enc.eval() if valid else self.txt_enc.train()

        ds_size = 0
        cum_loss = 0.0
        dataloader = self.train_loader if not valid else self.val_loader
        prog_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    
        for iteration, data in prog_bar:
            ids = data['ids'].to(self.device, dtype=torch.long)
            mask = data['mask'].to(self.device, dtype=torch.long)
            imgs = data['image'].to(self.device, dtype=torch.float)
            targets = data['target'].to(self.device, dtype=torch.long)
            bs = ids.size(0)

            img_embed = self.img_enc(imgs)
            txt_embed = self.txt_enc(ids, mask)
            loss = self.criterion(img_embed, txt_embed, targets)

            if not valid:
                loss /= SETTINGS['accum']
                loss.backward()

                if (iteration + 1) % SETTINGS['accum'] == 0:
                    self.img_optim.step()
                    self.img_optim.zero_grad()
                    self.txt_optim.step()
                    self.txt_optim.zero_grad()
                    if self.img_scheduler:
                        self.img_scheduler.step()
                    if self.txt_scheduler:
                        self.txt_scheduler.step()

            cum_loss += (loss.item() * bs)
            ds_size += bs
            ep_loss = cum_loss / ds_size

            prog_bar.set_postfix(Epoch=epoch, Train_Loss=ep_loss)
        gc.collect()

        return ep_loss

    def train_multiple_eps(self, num_epochs):
        start = time()
        img_enc_best = copy.deepcopy(self.img_enc.state_dict())
        txt_enc_best = copy.deepcopy(self.txt_enc.state_dict())
        history = defaultdict(list)
        ep_loss_best = np.inf

        for ep in range(1, num_epochs + 1):
            gc.collect()
            train_ep_loss = self.train_one_ep(ep)
            val_ep_loss = self.train_one_ep(ep, valid=True)

            history['Train Loss'].append(train_ep_loss)
            history['Valid Loss'].append(val_ep_loss)

            if val_ep_loss <= ep_loss_best:
                print(
                    f'Validation Loss Decreased from {ep_loss_best} to {val_ep_loss}')
                ep_loss_best = val_ep_loss
                img_enc_best = copy.deepcopy(self.img_enc.state_dict())
                txt_enc_best = copy.deepcopy(self.txt_enc.state_dict())
                img_enc_path = f"checkpoints/loss_{ep_loss_best:.4f}_ep{ep:.0f}_img_enc.pt"
                txt_enc_path = f"checkpoints/loss_{ep_loss_best:.4f}_ep{ep:.0f}_txt_enc.pt"
                torch.save(img_enc_best, img_enc_path)
                torch.save(txt_enc_best, txt_enc_path)
                print('Model saved successfully.')
            print('\n')

        total_time = time() - start
        print('Training completed in {:.0f} hours {:.0f} minutes {:.0f} seconds'.format(
            total_time // 3600, (total_time %
                                 3600) // 60, (total_time % 3600) % 60
        ))
        print(f'Best loss: {ep_loss_best:.4f}')
        self.img_enc.load_state_dict(img_enc_best)
        self.txt_enc.load_state_dict(txt_enc_best)

        return history

    def init_optim(self, optim_type='img'):
        return Lamb(self.img_enc.parameters() if optim_type == 'img' else self.txt_enc.parameters(),
                    lr=SETTINGS["eta"], weight_decay=SETTINGS["eta_decay"])

    def init_scheduler(self, sched_type='img'):
        return CosineAnnealingLR(self.img_optim if sched_type == 'img' else self.txt_optim,
                                 T_max=SETTINGS["T_max"], eta_min=SETTINGS["eta_min"])
