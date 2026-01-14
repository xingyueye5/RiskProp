import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from util import set_seed, save_checkpoint, compute_ap
from datasets import DashcamVideoDataset
from model import CCAFNetBaseline


def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    for frames, labels, meta in tqdm(loader, desc='train'):
        frames = frames.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(frames)  # [B, T]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())
    return running_loss / len(loader)


def val_epoch(model, loader, device):
    model.eval()
    y_true = []
    y_score = []
    with torch.no_grad():
        for frames, labels, meta in tqdm(loader, desc='val'):
            frames = frames.to(device)
            outputs = model(frames)
            y_true.append(labels.view(-1).cpu().numpy())
            y_score.append(outputs.view(-1).cpu().numpy())
    import numpy as np
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    ap = compute_ap(y_true, y_score)
    return ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--ann_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cuda', action='store_true')
    # Model hyperparameters
    parser.add_argument('--feat_dim', type=int, default=512)
    parser.add_argument('--rnn_hidden', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--cascade_stages', type=int, default=2)
    args = parser.parse_args()

    set_seed(42)
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    train_ds = DashcamVideoDataset(args.data_root, args.ann_file, seq_len=args.seq_len, step=1, mode='train')
    val_ds = DashcamVideoDataset(args.data_root, args.ann_file, seq_len=args.seq_len, step=args.seq_len, mode='val')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = CCAFNetBaseline(
        pretrained=True,
        feat_dim=args.feat_dim,
        rnn_hidden=args.rnn_hidden,
        num_layers=args.num_layers,
        cascade_stages=args.cascade_stages
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    best_ap = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        print(f'Epoch {epoch}')
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        print('Train loss:', train_loss)
        ap = val_epoch(model, val_loader, device)
        print('Val AP:', ap)
        if ap > best_ap:
            best_ap = ap
            save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                            os.path.join(args.save_dir, 'best.pth'))

if __name__ == '__main__':
    main()
