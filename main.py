import os 
import json
import argparse
from datetime import datetime as dt

import torch 
from torch import nn 
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils import random_seed, last_commit_msg, save_dependencies

MIN_VAL = 1e-6

# ------------------------
# Model 
# ------------------------

class Model(pl.LightningModule):

    # --------------------
    # Model Definition
    # --------------------

    def __init__(self, args):
        super().__init__()
        # args
        self.args = args

    def forward(self, batch):
        pass

    # -------------------
    # Training Definition
    # -------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.decay)
        return optimizer 

    def training_step(self, batch, batch_idx):
        aspect_label, senti_label = batch['aspect_label'].squeeze(-1), batch['senti_label'].squeeze(-1)
        aspect_pred, senti_pred = self.forward(batch)
        # loss calculation
        aspect_loss = self.aspect_criterion(aspect_pred, aspect_label)
        senti_loss = self.senti_criterion(senti_pred, senti_label)
        # logging
        self.log('train/aspect_loss', aspect_loss.item())
        self.log('train/senti_loss', senti_loss.item())
        # multiple loss
        loss = aspect_loss + senti_loss
        return loss 

    def validation_step(self, batch, batch_idx):
        pass


    def test_step(self, batch, batch_idx):
        pass
 
# ------------------------
# Dataset 
# ------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, mode, args):
        self.args = args
        self.mode = mode

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
 
# ------------------------
# Argument 
# ------------------------

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    # dataset
    p.add_argument('--data', type=str, default='Clothing',
                   help='dataset name')    
    # training
    p.add_argument('--epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('--batch_size', type=int, default=128,
                   help='number of epochs for train')
    p.add_argument('--lr', type=float, default=0.001,
                   help='initial learning rate')
    p.add_argument('--seed', type=int, default=42,
                   help='random seed for model training')
    p.add_argument('--max_size', type=int, default=64,
                   help='max length of records')
    # model
    p.add_argument('--embed_size', type=int, default=16,
                   help='embed size for items')
    # regularization
    p.add_argument('--dropout', type=float, default=0,
                   help='model dropout')
    p.add_argument('--decay', type=float, default=0)
    # fp16
    p.add_argument('--fp16', action='store_true',
                   help="Whether to use 16-bit float precision instead of 32-bit")
    # logging and output
    p.add_argument('--print_every', type=float, default=10,
                   help="print evaluate results every X epoch")
    p.add_argument('--ckpt_dir', type=str, default='',
                   help='checkpoint saving directory')
    # loading
    p.add_argument('--load_pretrained_weights', type=str, default=None,
                   help='checkpoint directory to load')
    return p.parse_args()

# ------------------------
# Main Function
# ------------------------

if __name__ == "__main__":
    args = parse_arguments()
    args.seed = random_seed(args.seed)

    # logging folder
    branch, commit = last_commit_msg()
    args.ckpt_dir = os.path.join('checkpoints', branch, f"{commit}_seed_{args.seed}", args.ckpt_dir, dt.now().strftime("%Y-%m-%d-%H-%M-%S"))

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    with open(os.path.join(args.ckpt_dir, "args.log"), "w") as f:
        f.write(json.dumps(vars(args), indent=2)) 
    save_dependencies(args.ckpt_dir)

    # dataset
    data_path = f"data/{args.data}"
    print(f"set ckpt as {args.ckpt_dir}, data path as {data_path}")
    
    train_dataset = Dataset(path=data_path, mode="train", args=args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32)
    val_loader = DataLoader(Dataset(path=data_path, mode="validation", args=args), batch_size=args.batch_size, shuffle=False, num_workers=32)
    test_loader = DataLoader(Dataset(path=data_path, mode="test", args=args), batch_size=args.batch_size, shuffle=False, num_workers=32)

    # model and training
    model = Model(args)
    early_stop_callback = EarlyStopping(monitor="validation/aspect_acc", patience=10, verbose=False, mode="max")
    trainer = pl.Trainer(default_root_dir=args.ckpt_dir, gpus=1, callbacks=[early_stop_callback], max_epochs=args.epochs)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader, ckpt_path='best')
