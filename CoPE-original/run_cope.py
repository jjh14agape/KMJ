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

import random
import numpy as np


ending_time = 1.
burnin_time = 0.0
alpha = 0.98
hidden_size = 128
n_neg_samples = 16

tbptt_len = 20
delta_coef = 0.


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round

def load(name):
    if name in {'wikipedia', 'lastfm', 'reddit'}:
        df, feats = load_jodie_data(f'data/{name}.csv')
    else:
        df, feats = load_recommendation_data(f'data/{name}_5.csv')
    return df, feats

def set_simple_logger(dataset):
    os.makedirs('log', exist_ok=True)

    log_file = f'log/{dataset}_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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

def set_checkpoint_paths(dataset):
    runtime_id = f"{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    checkpoint_root = './saved_checkpoints'
    best_model_root = './best_models'

    checkpoint_dir = os.path.join(checkpoint_root, runtime_id)
    best_model_dir = os.path.join(best_model_root, runtime_id)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    get_checkpoint_path = lambda epoch: os.path.join(
        checkpoint_dir, f'checkpoint-epoch-{epoch}.pth'
    )

    best_model_path = os.path.join(best_model_dir, 'best-model.pth')

    return get_checkpoint_path, best_model_path

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
    parser.add_argument('--early_stop', type=int, default=5, help='link or link_sign or sign')
    parser.add_argument('--tolerance', type=float, default=1e-3, help='tolerated marginal improvement for early stopper')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--fast_eval', action='store_true')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.cuda}')
    df, feats = load(args.dataset)
    n_users, n_items = df.iloc[:, :2].max() + 1

    set_random_seed(args.seed)

    logger = set_simple_logger(args.dataset)
    early_stopper = EarlyStopMonitor(max_round=args.early_stop , tolerance=args.tolerance) 

    get_checkpoint_path, best_model_path = set_checkpoint_paths(args.dataset)

    train_dl, valid_dl, test_dl = get_dataloaders(df, feats, device, ending_time, burnin_time, alpha)
    model = CoPE(n_users, n_items, hidden_size, n_neg_samples).to(device)
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    # for epoch in range(50):
    #     logger.info(f'epoch: {epoch}')


    #     val_mrr, test_mrr = train_one_epoch(model, optimizer, train_dl, delta_coef, tbptt_len, valid_dl, test_dl, True)

    #     logger.info(f'val_mrr: {val_mrr:.4f}')
    #     logger.info(f'test_mrr: {test_mrr:.4f}')

    #     if early_stopper.early_stop_check(val_mrr):
    #         logger.info(f"Early stopping triggered at epoch {epoch}")
    #         logger.info(f"Best epoch: {early_stopper.best_epoch}")
    #         break

    early_stopper = EarlyStopMonitor(
        max_round=args.early_stop,
        tolerance=args.tolerance
    )

    for epoch in range(args.epochs):
        val_mrr, val_rec10, val_pre10, val_ndcg10 = train_one_epoch(
            model, optimizer,
            train_dl,
            delta_coef,
            tbptt_len,
            valid_dl,
            fast_eval=args.fast_eval
        )

        logger.info(f'epoch: {epoch}')
        logger.info(f'val MRR: {val_mrr:.4f}')
        logger.info(f'val REC@10: {val_rec10:.4f}')
        logger.info(f'val PRE@10: {val_pre10:.4f}')
        logger.info(f'val NDCG@10: {val_ndcg10:.4f}')

        checkpoint_path = get_checkpoint_path(epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mrr': val_mrr,
            'args': vars(args),
        }, checkpoint_path)

        logger.info(f'Saved checkpoint: {checkpoint_path}')

        if early_stopper.early_stop_check(val_mrr):
            logger.info(
                f'No improvement over {early_stopper.max_round} epochs, stop training'
            )
            logger.info(f'Best epoch: {early_stopper.best_epoch}')
            break

    logger.info("\n***** Training complete. Loading best checkpoint. *****\n")

    best_checkpoint_path = get_checkpoint_path(early_stopper.best_epoch)

    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    logger.info(f'Loaded best checkpoint at epoch {early_stopper.best_epoch}')
    logger.info(f'Best validation MRR: {checkpoint["val_mrr"]:.4f}')

    logger.info("\n***** Testing best checkpoint *****\n")

    test_mrr, test_rec10, test_pre10, test_ndcg10 = evaluate_test(model, test_dl)

    logger.info(f'Final Test MRR: {test_mrr:.4f}')
    logger.info(f'Final Test REC@10: {test_rec10:.4f}')
    logger.info(f'Final Test PRE@10: {test_pre10:.4f}')
    logger.info(f'Final Test NDCG@10: {test_ndcg10:.4f}')

    torch.save(model.state_dict(), best_model_path)
    logger.info(f'Saved best model to: {best_model_path}')