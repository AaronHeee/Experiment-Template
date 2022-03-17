import os
import json
import argparse
import torch
import random
import numpy as np

from tqdm import tqdm
from datetime import datetime as dt
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from utils import random_seed, last_commit_msg
from models.model import Model

from collections import OrderedDict, defaultdict

TER = 10 # early stopping

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, mode, args):
        self.args = args
        self.mode = mode
       
    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        u = self.users[index]
        if self.mode == "train":
            return torch.LongTensor(c), torch.LongTensor(a), torch.LongTensor(c_), torch.LongTensor(a_), torch.LongTensor([u]), torch.LongTensor([i]), torch.LongTensor([j])
        else:
            return torch.LongTensor(c), torch.LongTensor(a), torch.LongTensor(c_), torch.LongTensor(a_), torch.LongTensor([u]), torch.LongTensor(t)


def evaluate(model, val_iter, args=None):
    res = defaultdict(list)

    with torch.no_grad():
        model.eval()

        for b, batch in tqdm(enumerate(val_iter)):
            c, a, c_, a_, u, t = batch
            c, a, c_, a_, u, t = c.cuda(), a.cuda(), c_.cuda(), a_.cuda(), u.cuda(), t.cuda()

            with autocast():
                ori_output = model(c, a, c_, a_, u) # (b, |I|)
                preds = ori_output.topk(k=max(args.K))[-1]

            for pred, trg in zip(preds, t):
                trgs = set(trg.tolist()) - {args.pad}
                for k in args.K:
                    ori_pred = set(pred[:k].tolist()) - {args.pad}
                    ori_acc = len(trgs & ori_pred) / len(trgs | ori_pred)
                    ori_prec = len(trgs & ori_pred) / len(ori_pred) if len(ori_pred) else 0
                    ori_rec = len(trgs & ori_pred) / len(trgs)
                    ori_f1 = 2 / ((1/ (ori_rec + 1e-8))+ (1/(ori_prec + 1e-8)))

                    res[f'ori_prec@{k}'].append(ori_prec)
                    res[f'ori_rec@{k}'].append(ori_rec)
                    res[f'ori_acc@{k}'].append(ori_acc)
                    res[f'ori_f1@{k}'].append(ori_f1)

                res['loss'].append(0)

        return OrderedDict({i: np.mean(res[i]) for i in res})

def train(e, model, optimizer, train_iter, args):
    model.train()
    total_loss = []
    tqdm_iter = tqdm(train_iter)
    for b, batch in enumerate(tqdm_iter):
        c, a, c_, a_, u, i, j = batch
        c, a, c_, a_, u, i, j =  c.cuda(), a.cuda(), c_.cuda(), a_.cuda(), u.cuda(), i.cuda(), j.cuda()

        optimizer.zero_grad()

        # generation loss
        with autocast():

            ori_output_i = model(c, a, c_, a_, u, i)
            ori_output_j = model(c, a, c_, a_, u, j)
            ori_loss = - F.logsigmoid(ori_output_i - ori_output_j).mean()

        if args.fp16:
            args.scaler.scale(ori_loss).backward()
        else:
            ori_loss.backward()

        clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        total_loss.append(ori_loss.data.item())

        mean_loss = np.mean(total_loss)
        tqdm_iter.set_description(f"[Epoch {e}] [loss: {mean_loss:.4f}]")


def main(data_path, args, pretrained_weights=None):

    ckpt_dir = os.path.join(args.ckpt_dir, f"{last_commit_msg()}_seed_{args.seed}", dt.now().strftime("%Y-%m-%d-%H-%M-%S"))
    terminate_cnt = 0

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    assert torch.cuda.is_available()

    print("[!] loading dataset...")
    train_dataset = Dataset(path=data_path, mode="train", args=args)
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_data = DataLoader(Dataset(path=data_path, mode="valid", args=args), batch_size=args.batch_size, shuffle=True)
    test_data = DataLoader(Dataset(path=data_path, mode="test", args=args), batch_size=args.batch_size, shuffle=True)

    args.item_size = train_dataset.item_size
    args.user_size = train_dataset.user_size
    args.pad = train_dataset.pad_token
    args.cate_pad = args.cate_pad = train_dataset.cate_pad_token
    args.attr_pad = args.attr_pad = train_dataset.attr_pad_token

    print("[!] Instantiating models...")
    model = Model(args).cuda()

    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    args.scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
        
    print(model)

    val_metrics = evaluate(model, valid_data, args)
    best_prec = val_metrics['ori_f1@10']
    print("[Epoch 0]" + " | ".join([m + ':' + f'{val_metrics[m]:.4f}'  for m in val_metrics]))

    test_metrics = evaluate(model, test_data, args) 
    print("[Epoch 0]" + " | ".join([m + ':' + f'{test_metrics[m]:.4f}'  for m in test_metrics]))

    for e in range(1, args.epochs+1):
        train(e, model, optimizer, train_data, args)

        if e % args.print_every == 0:

            # Val loss
            val_metrics = evaluate(model, valid_data, args)
            val_prec = val_metrics['ori_f1@10']
            print(f"[Epoch {e}]" + " | ".join([m + ':' + f'{val_metrics[m]:.4f}'  for m in val_metrics]))

            # Test loss
            test_metrics = evaluate(model, test_data, args) 
            print(f"[Epoch {e}]" + " | ".join([m + ':' + f'{test_metrics[m]:.4f}'  for m in test_metrics]))

            # Save the model if the validation loss is the best we've seen so far.
            if not best_prec or val_prec > best_prec:
                print("[!] saving model...")
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f'model_{e}.pt'))
                best_prec = val_prec

                with open(os.path.join(ckpt_dir, "test.log"), "w") as f:
                    test_metrics.update({'epoch': e})
                    f.write(json.dumps(test_metrics, indent=2))
                with open(os.path.join(ckpt_dir, "args.log"), "w") as f:
                    f.write(json.dumps(args, indent=2)) 
                terminate_cnt = 0
            else:
                terminate_cnt += 1
            
        # early stop
        if terminate_cnt == TER:
            break

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    # dataset
    p.add_argument('--data', type=str, default='steam',
                   help='steam')    
    # training
    p.add_argument('--epochs', type=int, default=1000,
                   help='number of epochs for train')
    p.add_argument('--batch_size', type=int, default=128,
                   help='number of epochs for train')
    p.add_argument('--lr', type=float, default=0.001,
                   help='initial learning rate')
    p.add_argument('--seed', type=int, default=None,
                   help='random seed for model training')
    p.add_argument('--grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    # model
    p.add_argument('--embed_size', type=int, default=32,
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
    p.add_argument('--ckpt_dir', type=str, default='experiments/test',
                   help='checkpoint saving directory')
    # loading
    p.add_argument('--load_pretrained_weights', type=str, default=None,
                   help='checkpoint directory to load')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    args.seed = random_seed(args.seed)
    print(args)
    args.K = eval(args.K)
    main(f"data/{args.data}", args, args.load_pretrained_weights)
