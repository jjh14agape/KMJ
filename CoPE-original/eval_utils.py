import numpy as np


def mrr(ranks):
    ranks = np.array(ranks)
    return (1 / ranks).mean()


def recall_at_k(ranks, k):
    ranks = np.array(ranks)
    return (ranks <= k).mean()

def precision_at_k(ranks, k):
    ranks = np.array(ranks)
    return (ranks <= k).mean() / k

def ndcg_at_k(ranks, k):
    ranks = np.array(ranks)

    # hit만 계산
    mask = ranks <= k

    ndcg = np.zeros_like(ranks, dtype=float)
    ndcg[mask] = 1 / np.log2(ranks[mask] + 1)

    return ndcg.mean()