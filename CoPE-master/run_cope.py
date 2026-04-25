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

import sys
import datetime
now = datetime.datetime.now()
import random

ending_time = 1.
burnin_time = 0.0
alpha = 0.98
hidden_size = 128
n_neg_samples = 16

tbptt_len = 20
delta_coef = 0.


def load(args):
    # if args.dataset in {'wikipedia', 'lastfm', 'reddit', 'mooc'}:
    #     df, feats = load_jodie_data(args.dataset, args.datapath, args)
    # else:
    #     df, feats = load_recommendation_data(args.dataset, args.datapath, args)

    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
        item2id, item_sequence_id, item_timediffs_sequence, 
        timestamp_sequence, feature_sequence, y_true] = load_network(args)

    # 시퀀스들이 같은 길이라고 가정
    df = pd.DataFrame({
        'user': user_sequence_id,
        'item': item_sequence_id, 
        'timestamp': timestamp_sequence
    })

    # Load Feature
    if args.dataset == "douban_movie" or args.dataset == "ml1m":
        if args.dataset == "douban_movie":
            user_features, item_features = load_feature(args, True, False, user2id, item2id)
        else:  # ml1m
            user_features, item_features = load_feature(args, True, True, user2id, item2id)
        feature_sequence = create_edge_features(df, user_features, item_features)

    return df, feature_sequence

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
    parser.add_argument('--dataset', default='douban_movie', type=str)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--early_stop', type=int, default=5, help='link or link_sign or sign')
    parser.add_argument('--tolerance', type=float, default=1e-3, help='tolerated marginal improvement for early stopper')
    parser.add_argument('--fast_eval', action='store_true', help='(default: false)')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.cuda}')

    set_random_seed(args.seed)



    args.datapath = "../dataset/%s/%s.csv" % (args.dataset, args.dataset)
    args.user_feature_path = "../dataset/%s/user_feat.csv" % (args.dataset)
    args.item_feature_path = "../dataset/%s/item_feat.csv" % (args.dataset)
    log_path = './log/'
    best_model_root = './best_models/'
    checkpoint_root = './saved_checkpoints/'

    # args.datapath = "./DyRec/dataset/%s/%s.csv" % (args.dataset, args.dataset)
    # args.user_feature_path = "./DyRec/dataset/%s/user_feat.csv" % (args.dataset)
    # args.item_feature_path = "./DyRec/dataset/%s/item_feat.csv" % (args.dataset)
    # log_path = 'DyRec/jodie-master/log/'
    # best_model_root = 'DyRec/jodie-master/best_models/'
    # checkpoint_root = 'DyRec/jodie-master/saved_checkpoints/'

    logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys.argv, now, log_path, checkpoint_root, best_model_root)
    
    df, feats = load(args)
    n_users, n_items = df.iloc[:, :2].max() + 1

    train_dl, valid_dl, test_dl = get_dataloaders(df, feats, device, ending_time, burnin_time, alpha)
    model = CoPE(n_users, n_items, hidden_size, n_neg_samples).to(device)
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    early_stopper = EarlyStopMonitor(max_round=args.early_stop , tolerance=args.tolerance) 
    
    mean_learning_time_per_epoch = []
    mean_inference_time_per_epoch1 = []
    mean_inference_time_per_epoch2 = []
    
    for epoch in range(args.epochs):
        val_mrr, val_rec1, val_rec5, val_rec10, val_rec20, val_pre1, val_pre5, val_pre10, val_pre20, val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20,  \
                test_mrr, test_rec1, test_rec5, test_rec10, test_rec20, test_pre1, test_pre5, test_pre10, test_pre20, test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20,  \
                    learning_time_per_epoch, inference_time_per_epoch1, inference_time_per_epoch2, train_peak_memory_mb, val_peak_memory_mb, test_peak_memory_mb = train_one_epoch(model, optimizer, train_dl, delta_coef, tbptt_len, valid_dl, test_dl, fast_eval=args.fast_eval)

        mean_learning_time_per_epoch.append(learning_time_per_epoch)
        mean_inference_time_per_epoch1.append(inference_time_per_epoch1)
        mean_inference_time_per_epoch2.append(inference_time_per_epoch2)

        logger.info('Emb Reset: {}'.format('False'))
        logger.info('epoch: {}'.format(epoch))
        # logger.info('Total loss in this epoch: {}'.format(total_loss))
        # 같은 메트릭끼리 한 줄에 탭으로 구분하여 출력
        logger.info('val MRR: {:.4f}'.format(val_mrr))
        logger.info('val REC:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_rec1, val_rec5, val_rec10, val_rec20))
        logger.info('val PRE:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_pre1, val_pre5, val_pre10, val_pre20))
        logger.info('val NDCG:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20))
        # logger.info('Trainig time: {} sec'.format(learning_time_per_epoch))
        # logger.info('Inference time: {} sec'.format(inference_time_per_epoch1))

        logger.info('test MRR:\t{:.4f}'.format(test_mrr))
        logger.info('test REC:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_rec1, test_rec5, test_rec10, test_rec20))
        logger.info('test PRE:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_pre1, test_pre5, test_pre10, test_pre20))
        logger.info('test NDCG:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20))
        logger.info('Average training time per epoch: {}'.format(np.mean(mean_learning_time_per_epoch)))
        logger.info('Average inference time per epoch (val): {}'.format(np.mean(mean_inference_time_per_epoch1)))
        logger.info('Average inference time per epoch (test): {}'.format(np.mean(mean_inference_time_per_epoch2)))
        # logger.info('Inference time (testing): {} sec'.format(inference_time))

        logger.info('Training peak memory: {:.2f} MB'.format(train_peak_memory_mb))
        logger.info('Inference peak memory (val): {:.2f} MB'.format(val_peak_memory_mb))
        logger.info('Inference peak memory (test): {:.2f} MB'.format(test_peak_memory_mb))


        
