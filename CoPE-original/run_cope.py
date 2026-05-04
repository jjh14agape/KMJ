import torch
from torch import nn, optim
from torch.nn import functional as F

import argparse

from trainer import *
from dataloader import *
from data_utils import *
from eval_utils import *
from trainer import *
from cope import CoPE

import logging
import os
from datetime import datetime


ending_time = 1.
burnin_time = 0.0
alpha = 0.98
hidden_size = 128
n_neg_samples = 16

tbptt_len = 20
delta_coef = 0.


def load(name):
    if name in {'wikipedia', 'lastfm', 'reddit'}:
        df, feats = load_jodie_data(f'data/{name}.csv')
    else:
        df, feats = load_recommendation_data(f'data/{name}_5.csv')
    return df, feats

def set_simple_logger():
    os.makedirs('log', exist_ok=True)

    log_file = f'log/train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 기존 handler 제거 (중복 방지)
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.cuda}')
    df, feats = load(args.dataset)
    n_users, n_items = df.iloc[:, :2].max() + 1

    logger = set_simple_logger()

    train_dl, valid_dl, test_dl = get_dataloaders(df, feats, device, ending_time, burnin_time, alpha)
    model = CoPE(n_users, n_items, hidden_size, n_neg_samples).to(device)
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    for epoch in range(50):
        logger.info(f'epoch: {epoch}')


        train_one_epoch(model, optimizer, train_dl, delta_coef, tbptt_len, valid_dl, test_dl, True)

        logger.info(f'val_mrr: {val_mrr:.4f}')
        logger.info(f'test_mrr: {test_mrr:.4f}')
