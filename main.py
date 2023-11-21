## Define Dependency

import os
import json

import torch
from torch import nn
from torch.utils.data import DataLoader

import random
from collections import defaultdict
from copy import deepcopy
from jsonargparse import CLI, Namespace

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

EPS = 1e-6
K = [1, 5, 10, 50]

## Define Dataset

class Data(object):
    def __init__(self, args):
        self.args = args
        ......

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, mode, args):
        self.data = data.split[mode]
        self.args = args
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ......
        return {'history': torch.LongTensor(h), 'label': pos, 'negative': neg}

## Define Model

class Model(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        # save hyperparameters
        self.save_hyperparameters()
        # args
        self.args = args
        # model
        self.model = ...
        # loss defintion
        self.loss = ......

    def forward(self, h, i):
        ......

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.decay)

    def eval_step(self, batch, mode='train'):
        .....
        # metric
        for k in K:
            self.log(f'{mode}/H@{k}', (rank < k).float().mean().item(), prog_bar=(k==10), on_epoch=True, on_step=False)

    def training_step(self, batch, batch_idx):
        h, l, n = batch['history'], batch['label'], batch['negative']
        logits_pos = self.forward(h, l)
        logits_neg = self.forward(h, n)
        loss = self.loss(logits_pos, torch.ones_like(logits_pos).to(logits_pos)) \
              + self.loss(logits_neg, torch.zeros_like(logits_neg).to(logits_neg))
        self.log(f'train/loss', loss.item())

        if self.current_epoch % self.args.print_every == 0:
            self.eval_step(batch, 'train')

        return loss

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, 'valid')

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, 'test')

## Define Training

def main(data: str = 'ml-1m', data_dir: str = '../data',
         epochs: int = 200, batch_size: int = 256, lr: float = 0.001, decay: float = 0,
         max_len: int = 128, embed_size: int = 32, alpha: float = 1,
         print_every: int = 10, ckpt_dir: str = 'test',
         load_pretrained_weights: str = None, test_only: bool = False, max_minutes: int = 30):

    # args
    args = Namespace(**locals())
    print(args)

    # dataloader
    data = Data(args=args)
    train_loader = DataLoader(Dataset(data=data, mode="train", args=args), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Dataset(data=data, mode="valid", args=args), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(Dataset(data=data, mode="test", args=args), batch_size=args.batch_size, shuffle=False, num_workers=0)

    # update args
    args.n_users = data.n_users
    args.n_items = data.n_items
    args.pad_token = args.n_items

    # model and trainer
    model = Model(args)
    print(model)
    early_stop_callback = EarlyStopping(monitor="valid/H@10", patience=100, verbose=False, mode="max")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="valid/H@10", mode='max')
    trainer = pl.Trainer(default_root_dir=args.ckpt_dir, accelerator='auto', max_epochs=args.epochs, max_time={'minutes': args.max_minutes}, gradient_clip_val=5.0, callbacks=[early_stop_callback, checkpoint_callback], check_val_every_n_epoch=args.print_every)

    # mode fit or test
    if args.test_only:
        model.load_state_dict(torch.load(args.load_pretrained_weights)['state_dict'])
        trainer.test(model=model, dataloaders=test_loader)
    else:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(model=model, dataloaders=test_loader, ckpt_path='best')

if __name__ == "__main__":
    CLI(main)
