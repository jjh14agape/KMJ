# -*- coding: utf-8 -*
# from __future__ import division
# import numpy as np
# from collections import defaultdict
# from sklearn.preprocessing import scale
# import argparse
# import torch
# import random
# import os
# import logging

# # Load data
# def load_network(args, time_scaling=True):
#     '''
#     The dataset should be in the following format:
#     Each line should be: user, item, timestamp, state label, array of features.
#     Timestamp should be in cardinal format (not in datetime).
#     State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
#     Feature list can be as long as desired. It should be at least 1 dimensional. If there are no features, use 0 for all interactions.
#     '''

#     dataset = args.dataset
#     datapath = args.datapath

#     user_sequence = []
#     item_sequence = []
#     label_sequence = []
#     feature_sequence = []
#     timestamp_sequence = []
#     start_timestamp = None
#     y_true_labels = []

#     print("\n***** Loading %s dataset from file: %s *****" % (dataset, datapath))
#     f = open(datapath,"r")
#     if dataset != 'ml1m' and dataset != 'douban_movie':
#         f.readline()
#     '''for cnt, l in enumerate(f):
#         # format: user, item, timestamp, state label, feature list
#         ls = l.strip().split(",")
#         user_sequence.append(ls[0])
#         item_sequence.append(ls[1])
#         if start_timestamp is None:
#             start_timestamp = float(ls[2])
#         timestamp_sequence.append(float(ls[2]) - start_timestamp) 
#         y_true_labels.append(int(ls[3]))
#         feature_sequence.append(list(map(float, ls[4:])))
#     f.close()'''
#     for cnt, l in enumerate(f):
#         if dataset == 'ml1m' or dataset == 'douban_movie': 
#             ls = l.strip().split("\t")
#             user_sequence.append(ls[0])
#             item_sequence.append(ls[1])
#             if start_timestamp is None:
#                 start_timestamp = float(ls[3])
#             timestamp_sequence.append(float(ls[3]) - start_timestamp) 
#             # y_true_labels.append(int(ls[2]))
#             # 여기 수정해야함 feature.csv 에서 가져오는 걸로
#             # feature_sequence.append(list(map(float, ls[4:])))

#         else: # format: user, item, timestamp, state label, feature list
#             ls = l.strip().split(",")
#             user_sequence.append(ls[0])
#             item_sequence.append(ls[1])
#             if start_timestamp is None:
#                 start_timestamp = float(ls[2])
#             timestamp_sequence.append(float(ls[2]) - start_timestamp) 
#             # y_true_labels.append(int(ls[3]))
#             feature_sequence.append(list(map(float, ls[4:])))
#     f.close()

#     user_sequence = np.array(user_sequence)
#     item_sequence = np.array(item_sequence)
#     timestamp_sequence = np.array(timestamp_sequence)

#     print("Formatting item sequence")
#     nodeid = 0
#     item2id = {}
#     item_timedifference_sequence = []
#     item_current_timestamp = defaultdict(float)
#     for cnt, item in enumerate(item_sequence):
#         if item not in item2id:
#             item2id[item] = nodeid
#             nodeid += 1
#         timestamp = timestamp_sequence[cnt]
#         item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
#         item_current_timestamp[item] = timestamp
#     num_items = len(item2id)
#     item_sequence_id = [item2id[item] for item in item_sequence]

#     print("Formatting user sequence")
#     nodeid = 0
#     user2id = {}
#     user_timedifference_sequence = []
#     user_current_timestamp = defaultdict(float)
#     user_previous_itemid_sequence = []
#     user_latest_itemid = defaultdict(lambda: num_items)
#     for cnt, user in enumerate(user_sequence):
#         if user not in user2id:
#             user2id[user] = nodeid
#             nodeid += 1
#         timestamp = timestamp_sequence[cnt]
#         user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
#         user_current_timestamp[user] = timestamp
#         user_previous_itemid_sequence.append(user_latest_itemid[user])
#         user_latest_itemid[user] = item2id[item_sequence[cnt]]
#     num_users = len(user2id)
#     user_sequence_id = [user2id[user] for user in user_sequence]

#     # for time-decay GCN
#     timedifference_sequence_for_adj = (np.array(timestamp_sequence)/(3600*24)).astype(int)  # based on day

#     if time_scaling:
#         print("Scaling timestamps")
#         user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
#         item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

#     print("***** dataset loading completed *****\n")
#     return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence,
#             item2id, item_sequence_id, item_timedifference_sequence,
#             timestamp_sequence, feature_sequence, timedifference_sequence_for_adj]


# def set_random_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ['PYTHONHASHSEED'] = str(seed)

# def load_feature(args, item_feat_flag, user_feat_flag, item2id, user2id):
#     if (item_feat_flag == True) and (user_feat_flag == True):       
#         user_features = load_user_feat(args.user_feature_path, user2id)
#         item_features = load_item_feat(args.item_feature_path, item2id)
#     elif (item_feat_flag == True) and (user_feat_flag == False):
#         user_features = None
#         item_features = load_item_feat(args.item_feature_path, item2id)
        
#     return user_features, item_features

# def load_user_feat(user_feature_path, user2id):
#     user_features = {}
#     with open(user_feature_path, "r") as f:
#         # f.readline()  # 첫 줄 스킵
#         for user_id, feat in enumerate(f):
#             user_features[user2id[str(user_id)]] = list(map(int, feat.strip().split()))
#     return user_features

# def load_item_feat(item_feature_path, item2id):
#     item_features = {}
#     with open(item_feature_path, "r") as f:
#         # f.readline()  # 첫 줄 스킵
#         for item_id, feat in enumerate(f):
#             item_features[item2id[str(item_id)]] = list(map(int, feat.strip().split()))
#     return item_features


# def str2bool(value):
#     """Converts string to boolean"""
#     if value.lower() in ('yes', 'true', 't', '1'):
#         return True
#     elif value.lower() in ('no', 'false', 'f', '0'):
#         return False
#     else:
#         raise ValueError("Boolean value expected")
    

# def set_up_logger(args, sys_argv, now, log_path, checkpoint_root, best_model_root):
#     runtime_id = '{}-{}-{}-{}-{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute,  now.second, args.dataset)
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)
#     file_path = log_path + '{}/{}.log'.format(args.dataset, runtime_id) 
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     fh = logging.FileHandler(file_path)
#     fh.setLevel(logging.DEBUG)
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.WARN)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     fh.setFormatter(formatter)
#     ch.setFormatter(formatter)
#     logger.addHandler(fh)
#     logger.addHandler(ch)
#     logger.info('Create log file at {}'.format(file_path))
#     logger.info('Command line executed: python ' + ' '.join(sys_argv))
#     logger.info('Full args parsed:')
#     logger.info(args)
#     logger.info('cold-start')

#     # checkpoint_root = './saved_checkpoints/'
#     checkpoint_dir = checkpoint_root + runtime_id + '/'
#     # best_model_root = './best_models/'
#     best_model_dir = best_model_root + runtime_id + '/'
#     if not os.path.exists(checkpoint_root):
#         os.mkdir(checkpoint_root)
#         logger.info('Create directory {}'.format(checkpoint_root))
#     if not os.path.exists(best_model_root):
#         os.mkdir(best_model_root)
#         logger.info('Create directory'.format(best_model_root))
#     os.mkdir(checkpoint_dir)
#     os.mkdir(best_model_dir)
#     logger.info('Create checkpoint directory {}'.format(checkpoint_dir))
#     logger.info('Create best model directory {}'.format(best_model_dir))

#     get_checkpoint_path = lambda epoch: (checkpoint_dir + 'checkpoint-epoch-{}.pth'.format(epoch))
#     best_model_path = best_model_dir + 'best-model.pth'

#     return logger, get_checkpoint_path, best_model_path


from __future__ import division
import numpy as np
import random
import sys
import operator
import copy
from collections import defaultdict
import os, re
import argparse
from sklearn.preprocessing import scale
import torch
import logging

# LOAD THE DATASET
def load_network(args, time_scaling=True):
    dataset = args.dataset
    datapath = args.datapath

    user_sequence = []
    item_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    rating_sequence = []
    start_timestamp = None
    y_true_labels = []

    print("\n\n**** Loading %s network from file: %s ****" % (dataset, datapath))
    f = open(datapath,"r")
    f.readline()
    for cnt, l in enumerate(f): # FORMAT: user_id,item_id,rating,timestamp
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        rating_sequence.append(float(ls[2]))
        if start_timestamp is None:
            start_timestamp = float(ls[3])
        timestamp_sequence.append(float(ls[3]) - start_timestamp)
        feature_sequence.append(list(map(float,ls[4:])))

    f.close()

    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)
    rating_sequence = np.array(rating_sequence)
    timestamp_sequence = np.array(timestamp_sequence)
    sign_sequence = np.where(rating_sequence >= 4, 1, -1)

    print("Formating item sequence")
    nodeid = 1
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    print("Formating user sequence")
    # nodeid = 1
    nodeid = 0
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: num_items)
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    # for time-decay GCN
    timedifference_sequence_for_adj = (np.array(timestamp_sequence)/(3600*24)).astype(int)  # based on day

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    # rating value => one-hot vector로 변환
    rating_onehot = np.zeros((len(rating_sequence), 5))
    rating_onehot[np.arange(len(rating_sequence)), (rating_sequence - 1).astype(int)] = 1
    rating_sequence = rating_onehot  # (num_interactions, 5)
    
    print("*** Network loading completed ***\n\n")
    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, feature_sequence, rating_sequence, sign_sequence, y_true_labels, timedifference_sequence_for_adj]


# def set_random_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed)
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)

def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# def load_feature(args, item_feat_flag, user_feat_flag, item2id, user2id):
#     if (item_feat_flag == True) and (user_feat_flag == True):       
#         user_features = load_user_feat(args.user_feature_path, user2id)
#         item_features = load_item_feat(args.item_feature_path, item2id)
#     elif (item_feat_flag == True) and (user_feat_flag == False):
#         user_features = None
#         item_features = load_item_feat(args.item_feature_path, item2id)
        
#     return user_features, item_features

# def load_user_feat(user_feature_path, user2id):
#     user_features = {}
#     with open(user_feature_path, "r") as f:
#         for user_id, feat in enumerate(f):
#             user_features[user2id[str(user_id)]] = list(map(int, feat.strip().split()))
#     return user_features

# def load_item_feat(item_feature_path, item2id):
#     item_features = {}
#     with open(item_feature_path, "r") as f:
#         for item_id, feat in enumerate(f):
#             item_features[item2id[str(item_id)]] = list(map(int, feat.strip().split()))
#     return item_features


def str2bool(value):
    """Converts string to boolean"""
    if value.lower() in ('yes', 'true', 't', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise ValueError("Boolean value expected")
    

def set_up_logger(args, sys_argv, now, log_path, checkpoint_root, best_model_root):
    runtime_id = '{}-{}-{}-{}-{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute,  now.second, args.dataset)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    directory = log_path + '{}/'.format(args.dataset) 
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    file_path = log_path + '{}/{}.log'.format(args.dataset, runtime_id) 
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)

    checkpoint_dir = checkpoint_root + runtime_id + '/'
    best_model_dir = best_model_root + runtime_id + '/'
    if not os.path.exists(checkpoint_root):
        os.makedirs(checkpoint_root, exist_ok=True)
        logger.info('Create directory {}'.format(checkpoint_root))
    if not os.path.exists(best_model_root):
        os.makedirs(best_model_root, exist_ok=True)
        logger.info('Create directory'.format(best_model_root))
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    logger.info('Create checkpoint directory {}'.format(checkpoint_dir))
    logger.info('Create best model directory {}'.format(best_model_dir))

    get_checkpoint_path = lambda epoch: (checkpoint_dir + 'checkpoint-epoch-{}.pth'.format(epoch))
    best_model_path = best_model_dir + 'best-model.pth'

    return logger, get_checkpoint_path, best_model_path


def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)    
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2) 
        return allocated, reserved, peak
    return 0, 0, 0

def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()