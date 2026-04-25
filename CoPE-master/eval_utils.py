import numpy as np
import math

def mrr(ranks):
    ranks = np.array(ranks)
    return (1 / ranks).mean()


def recall_at_k(ranks, k):
    ranks = np.array(ranks)
    return (ranks <= k).mean()


def precision_at_k(ranks, k):
    pre = sum(np.array(ranks) <= k) / (len(ranks) * k)
    return pre

def ndcg_at_k(ranks, k):
    ndcgs = []
    for r in ranks:
        if r <= k:
            dcg = 1.0 / math.log2(r + 1)
            idcg = 1.0  # 정답이 1등일 때
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)
    return np.mean(ndcgs)