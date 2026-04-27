import time
from copy import deepcopy

import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F

from cope import CoPE
from model_utils import *
from eval_utils import *
from data_utils import get_gpu_memory_usage, reset_gpu_memory_stats

# epoch, model, optimizer, train_dl, delta_coef, tbptt_len, valid_dl, test_dl, False
def train_one_epoch(model, optimizer, train_dl, delta_coef=1e-5, tbptt_len=20,
                    valid_dl=None, test_dl=None, fast_eval=True, adaptation=False, adaptation_lr=1e-4):
    # print(str(next(model.parameters()).device))  # 'cuda:0' 또는 'cpu'
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats(device=device)
    torch.cuda.synchronize(device=device)
    epoch_start_time = time.time()
    
    last_xu, last_xi = model.get_init_states()
    loss_pp = 0.
    loss_norm = 0.
    optimizer.zero_grad()
    model.train()
    counter = 0
    pbar = tqdm.tqdm(train_dl)
    cum_loss = 0.
    for i, batch in enumerate(pbar):
        t, dt, adj, i2u_adj, u2i_adj, users, items = batch
        step_loss, delta_norm, last_xu, last_xi, *_ = model.propagate_update_loss(adj, dt, last_xu, last_xi, i2u_adj, u2i_adj, users, items)
        loss_pp += step_loss
        loss_norm += delta_norm
        counter += 1
    
        if (counter % tbptt_len) == 0 or i == (len(train_dl) - 1):
            total_loss = (loss_pp + loss_norm * delta_coef) / counter
            total_loss.backward()
            optimizer.step()
            cum_loss += total_loss.item()
            pbar.set_description(f"Loss={cum_loss/i:.4f}")
            last_xu = last_xu.detach()
            last_xi = last_xi.detach()
            optimizer.zero_grad()
            loss_pp = 0.
            loss_norm = 0.
            counter = 0
    pbar.close()
    
    # ✅ 학습 종료 시 GPU 동기화
    torch.cuda.synchronize(device=device)
    learning_time_per_epoch = time.time() - epoch_start_time

    # ✅ 학습 피크 메모리 측정
    train_peak_memory_mb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)

    
    
    if fast_eval:
        val_mrr, val_rec1, val_rec5, val_rec10, val_rec20, val_pre1, val_pre5, val_pre10, val_pre20, val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20,  \
                test_mrr, test_rec1, test_rec5, test_rec10, test_rec20, test_pre1, test_pre5, test_pre10, test_pre20, test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20, \
                    inference_time_per_epoch1, inference_time_per_epoch2, val_peak_memory_mb, test_peak_memory_mb = rollout_evaluate_fast(model, valid_dl, test_dl, last_xu.detach(), last_xi.detach())
    else:
        val_mrr, val_rec1, val_rec5, val_rec10, val_rec20, val_pre1, val_pre5, val_pre10, val_pre20, val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20,  \
                test_mrr, test_rec1, test_rec5, test_rec10, test_rec20, test_pre1, test_pre5, test_pre10, test_pre20, test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20, \
                    inference_time_per_epoch1, inference_time_per_epoch2, val_peak_memory_mb, test_peak_memory_mb = rollout_evaluate(model, train_dl, valid_dl, test_dl)    

    return val_mrr, val_rec1, val_rec5, val_rec10, val_rec20, val_pre1, val_pre5, val_pre10, val_pre20, val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20,  \
                test_mrr, test_rec1, test_rec5, test_rec10, test_rec20, test_pre1, test_pre5, test_pre10, test_pre20, test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20,  \
                    learning_time_per_epoch, inference_time_per_epoch1, inference_time_per_epoch2, train_peak_memory_mb, val_peak_memory_mb, test_peak_memory_mb


def rollout_evaluate_fast(model, valid_dl, test_dl, train_xu, train_xi):
    # ✅ 추론 시작 전 메모리 리셋
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats(device=device)
    
    # ✅ 추론 시작 전 GPU 동기화
    torch.cuda.synchronize(device=device)
    validate_start_time = time.time()

    valid_xu, valid_xi, valid_ranks = rollout(valid_dl, model, train_xu, train_xi)
    
    torch.cuda.synchronize(device=device)
    inference_time_per_epoch1 = time.time()-validate_start_time

    # ✅ validation 피크 메모리 측정
    val_peak_memory_mb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)

    val_mrr = mrr(valid_ranks)
    val_rec1 = recall_at_k(valid_ranks, 1)
    val_rec5 = recall_at_k(valid_ranks, 5)
    val_rec10 = recall_at_k(valid_ranks, 10)
    val_rec20 = recall_at_k(valid_ranks, 20)
    val_pre1 = precision_at_k(valid_ranks, 1)
    val_pre5 = precision_at_k(valid_ranks, 5)
    val_pre10 = precision_at_k(valid_ranks, 10) 
    val_pre20 = precision_at_k(valid_ranks, 20)
    val_ndcg1 = ndcg_at_k(valid_ranks, 1) 
    val_ndcg5 = ndcg_at_k(valid_ranks, 5)
    val_ndcg10 = ndcg_at_k(valid_ranks, 10)
    val_ndcg20 = ndcg_at_k(valid_ranks, 20)

    test_xu = train_xu.detach().clone()  # 훈련 종료 시점 상태로 리셋
    test_xi = train_xi.detach().clone()
    
    # ✅ test 시작 전 메모리 리셋
    torch.cuda.reset_peak_memory_stats(device=device)

    torch.cuda.synchronize(device=device)
    test_start_time = time.time()
    # print(f"------- Valid MRR: {mrr(valid_ranks):.4f} Recall@10: {recall_at_k(valid_ranks, 10):.4f}")
    # _u, _i, test_ranks = rollout(test_dl, model, test_xu, test_xi)
    _u, _i, test_ranks = rollout(test_dl, model, valid_xu, valid_xi)
    # print(f"=======  Test MRR: {mrr(test_ranks):.4f} Recall@10: {recall_at_k(test_ranks, 10):.4f}")
    torch.cuda.synchronize(device=device)
    inference_time_per_epoch2 = time.time() - test_start_time

    # ✅ test 피크 메모리 측정
    test_peak_memory_mb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)


    test_mrr = mrr(test_ranks)
    test_rec1 = recall_at_k(test_ranks, 1)
    test_rec5 = recall_at_k(test_ranks, 5)
    test_rec10 = recall_at_k(test_ranks, 10)
    test_rec20 = recall_at_k(test_ranks, 20)
    test_pre1 = precision_at_k(test_ranks, 1)
    test_pre5 = precision_at_k(test_ranks, 5)
    test_pre10 = precision_at_k(test_ranks, 10) 
    test_pre20 = precision_at_k(test_ranks, 20)
    test_ndcg1 = ndcg_at_k(test_ranks, 1) 
    test_ndcg5 = ndcg_at_k(test_ranks, 5)
    test_ndcg10 = ndcg_at_k(test_ranks, 10)
    test_ndcg20 = ndcg_at_k(test_ranks, 20)
    return val_mrr, val_rec1, val_rec5, val_rec10, val_rec20, val_pre1, val_pre5, val_pre10, val_pre20, val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20, \
        test_mrr, test_rec1, test_rec5, test_rec10, test_rec20, test_pre1, test_pre5, test_pre10, test_pre20, test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20, \
        inference_time_per_epoch1, inference_time_per_epoch2, val_peak_memory_mb, test_peak_memory_mb

def rollout_evaluate(model, train_dl, valid_dl, test_dl):
    device = next(model.parameters()).device
    # ✅ 추론 시작 전 메모리 리셋
    torch.cuda.reset_peak_memory_stats(device=device)

    train_xu, train_xi, train_ranks = rollout(train_dl, model, *model.get_init_states())
    
    # print(f"Train MRR: {mrr(train_ranks):.4f} Recall@10: {recall_at_k(train_ranks, 10):.4f}")
    # ✅ 추론 시작 전 GPU 동기화
    torch.cuda.synchronize(device=device)
    validate_start_time = time.time()
    valid_xu, valid_xi, valid_ranks = rollout(valid_dl, model, train_xu, train_xi)
    torch.cuda.synchronize(device=device)
    inference_time_per_epoch1 = time.time()-validate_start_time
    # print(f"Valid MRR: {mrr(valid_ranks):.4f} Recall@10: {recall_at_k(valid_ranks, 10):.4f}")
    
    # ✅ validation 피크 메모리 측정
    val_peak_memory_mb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)

    val_mrr = mrr(valid_ranks)
    val_rec1 = recall_at_k(valid_ranks, 1)
    val_rec5 = recall_at_k(valid_ranks, 5)
    val_rec10 = recall_at_k(valid_ranks, 10)
    val_rec20 = recall_at_k(valid_ranks, 20)
    val_pre1 = precision_at_k(valid_ranks, 1)
    val_pre5 = precision_at_k(valid_ranks, 5)
    val_pre10 = precision_at_k(valid_ranks, 10) 
    val_pre20 = precision_at_k(valid_ranks, 20)
    val_ndcg1 = ndcg_at_k(valid_ranks, 1) 
    val_ndcg5 = ndcg_at_k(valid_ranks, 5)
    val_ndcg10 = ndcg_at_k(valid_ranks, 10)
    val_ndcg20 = ndcg_at_k(valid_ranks, 20)

    test_xu = train_xu.detach().clone()  # 훈련 종료 시점 상태로 리셋
    test_xi = train_xi.detach().clone()
    
    # ✅ test 시작 전 메모리 리셋
    torch.cuda.reset_peak_memory_stats(device=device)
    torch.cuda.synchronize(device=device)
    test_start_time = time.time()
    _u, _i, test_ranks = rollout(test_dl, model, test_xu, test_xi)
    torch.cuda.synchronize(device=device)
    inference_time_per_epoch2 = time.time()-test_start_time

    # ✅ test 피크 메모리 측정
    test_peak_memory_mb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)

    test_mrr = mrr(test_ranks)
    test_rec1 = recall_at_k(test_ranks, 1)
    test_rec5 = recall_at_k(test_ranks, 5)
    test_rec10 = recall_at_k(test_ranks, 10)
    test_rec20 = recall_at_k(test_ranks, 20)
    test_pre1 = precision_at_k(test_ranks, 1)
    test_pre5 = precision_at_k(test_ranks, 5)
    test_pre10 = precision_at_k(test_ranks, 10) 
    test_pre20 = precision_at_k(test_ranks, 20)
    test_ndcg1 = ndcg_at_k(test_ranks, 1) 
    test_ndcg5 = ndcg_at_k(test_ranks, 5)
    test_ndcg10 = ndcg_at_k(test_ranks, 10)
    test_ndcg20 = ndcg_at_k(test_ranks, 20)

    # print(f"Test MRR: {mrr(test_ranks):.4f} Recall@10: {recall_at_k(test_ranks, 10):.4f}")
    return val_mrr, val_rec1, val_rec5, val_rec10, val_rec20, val_pre1, val_pre5, val_pre10, val_pre20, val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20, \
        test_mrr, test_rec1, test_rec5, test_rec10, test_rec20, test_pre1, test_pre5, test_pre10, test_pre20, test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20, \
        inference_time_per_epoch1, inference_time_per_epoch2, val_peak_memory_mb, test_peak_memory_mb

def rollout(dl, model, last_xu, last_xi):
    model.eval()
    ranks = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dl, position=0):
            t, dt, adj, i2u_adj, u2i_adj, users, items = batch
            prop_user, prop_item, last_xu, last_xi = model.propagate_update(adj, dt, last_xu, last_xi, i2u_adj, u2i_adj)
            rs = compute_rank(model, prop_user, prop_item, users, items)
            ranks.extend(rs)
    return last_xu, last_xi, ranks


def compute_rank(model: CoPE, xu, xi, users, items):
    xu = torch.cat([xu, model.user_states], 1)
    xi = torch.cat([xi, model.item_states], 1)
    xu = F.embedding(users, xu)
    scores = model.compute_pairwise_scores(xu, xi)
    ranks = []
    for line, i in zip(scores, items):
        r = (line >= line[i]).sum().item()
        ranks.append(r)
    return ranks

