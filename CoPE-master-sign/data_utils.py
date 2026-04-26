import numpy as np
import pandas as pd
import logging
import os, re
from collections import defaultdict
from sklearn.preprocessing import scale
import torch 

# def check_dataframe(df):
#     assert df.iloc[:, 0].max() + 1 == df.iloc[:, 0].nunique()
#     assert df.iloc[:, 1].max() + 1 == df.iloc[:, 1].nunique()
#     assert (df.iloc[:, 2].diff().iloc[1:] >= 0).all()

def check_dataframe(df):
    assert df.iloc[:, 0].max() + 1 == df.iloc[:, 0].nunique()
    assert df.iloc[:, 1].max() + 1 == df.iloc[:, 1].nunique()
    assert (df['timestamp'].diff().iloc[1:] >= 0).all()

    
def load_jodie_data(dataset, datapath, args):
    with open(datapath) as fh:
        interactions = []
        features = []
        for i, line in enumerate(fh):
            if i == 0:
                header = line.strip().split(',')
                continue
            uid, iid, ts, state, *feat = line.strip().split(',')
            interactions.append([int(uid), int(iid), float(ts), int(state)])
            features.append([float(v) for v in feat])
    df = pd.DataFrame(interactions, columns=header[:-1]) 
    features = np.asarray(features)
    check_dataframe(df)
    return df, features


def load_recommendation_data(dataset, datapath, args):
    if 'douban_movie' in datapath or 'ml1m' in datapath: # FORMAT: user, item, rating, timestamp, state
        # 탭으로 구분되고 헤더가 없는 경우
        df = pd.read_csv(datapath, sep='\t', header=None)
        # 컬럼명이 다를 수 있으므로 표준화
        df = df.iloc[:, [0, 1, 3]]
        df.columns = ['user', 'item', 'timestamp']
    else:
        df = pd.read_csv(datapath, header=0) # 첫번째 헤더 행 pass
        df.columns = ['user', 'item', 'timestamp']
    df.iloc[:, :2] -= 1
    df.timestamp -= df.timestamp.min()
    check_dataframe(df)
    if 'douban_movie' in datapath:
        features = load_feature(args, True, False)
    elif 'ml1m' in datapath:
        features = load_feature(args, True, True)
    else:
        features = np.zeros((len(df), 1))
    return df, features


def data_split(train_proportion, df, feats):
    df = df.copy()
    num_interactions = len(df)
    train_end_idx = validation_start_idx = int(num_interactions * train_proportion)
    test_start_idx = int(num_interactions * (train_proportion + .1))
    test_end_idx = int(num_interactions * (train_proportion + .2))
    df_train = df.iloc[:train_end_idx]
    df_valid = df.iloc[validation_start_idx:test_start_idx]
    df_test = df.iloc[test_start_idx:test_end_idx]
    feats_train = feats[:train_end_idx]
    feats_valid = feats[validation_start_idx:test_start_idx]
    feats_test = feats[test_start_idx:test_end_idx]
    return df_train, df_valid, df_test, feats_train, feats_valid, feats_test


def recommendation_to_jodie(in_fname, out_fname):
    df, _ = load_recommendation_data(in_fname)
    df.columns = ['user_id', 'item_id', 'timestamp']
    df['state_label'] = 0
    df['separated_list_of_features'] = 0.0
    df.to_csv(out_fname, index=False)


#mj
def set_up_logger(args, sys_argv, now, log_path, checkpoint_root, best_model_root):
    runtime_id = '{}-{}-{}-{}-{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute,  now.second, args.dataset)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    directory = log_path + '{}/'.format(args.dataset) 
    if not os.path.exists(directory):
        os.makedirs(directory)
        
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

    # checkpoint_root = './saved_checkpoints/'
    checkpoint_dir = checkpoint_root + runtime_id + '/'
    # best_model_root = './best_models/'
    best_model_dir = best_model_root + runtime_id + '/'
    if not os.path.exists(checkpoint_root):
        os.mkdir(checkpoint_root)
        logger.info('Create directory {}'.format(checkpoint_root))
    if not os.path.exists(best_model_root):
        os.mkdir(best_model_root)
        logger.info('Create directory'.format(best_model_root))
    os.mkdir(checkpoint_dir)
    os.mkdir(best_model_dir)
    logger.info('Create checkpoint directory {}'.format(checkpoint_dir))
    logger.info('Create best model directory {}'.format(best_model_dir))

    get_checkpoint_path = lambda epoch: (checkpoint_dir + 'checkpoint-epoch-{}.pth'.format(epoch))
    best_model_path = best_model_dir + 'best-model.pth'

    return logger, get_checkpoint_path, best_model_path

def load_feature(args, item_feat_flag, user_feat_flag, user2id, item2id):
    if (item_feat_flag == True) and (user_feat_flag == True):       
        user_features = load_user_feat(args.user_feature_path, user2id)
        item_features = load_item_feat(args.item_feature_path, item2id)
    elif (item_feat_flag == True) and (user_feat_flag == False):
        user_features = None
        item_features = load_item_feat(args.item_feature_path, item2id)
        
    return user_features, item_features

def load_user_feat(user_feature_path, user2id):
    user_features = {}
    with open(user_feature_path, "r") as f:
        # f.readline()  # 첫 줄 스킵
        for user_id, feat in enumerate(f):
            user_features[user2id[str(user_id)]] = list(map(int, feat.strip().split()))
    return user_features

def load_item_feat(item_feature_path, item2id):
    item_features = {}
    with open(item_feature_path, "r") as f:
        # f.readline()  # 첫 줄 스킵
        for item_id, feat in enumerate(f):
            item_features[item2id[str(item_id)]] = list(map(int, feat.strip().split()))
    return item_features


# def load_network(args, time_scaling=True):
#     '''
#     This function loads the input network.

#     The network should be in the following format:
#     One line per interaction/edge.
#     Each line should be: user, item, timestamp, state label, array of features.
#     Timestamp should be in cardinal format (not in datetime).
#     State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
#     Feature list can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions. 
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

#     print("\n\n**** Loading %s network from file: %s ****" % (dataset, datapath))
#     f = open(datapath,"r")
#     if dataset != 'ml1m' and dataset != 'douban_movie':
#         f.readline()  # only skip header if not ml1m or douban_movie
#     for cnt, l in enumerate(f):
#         if dataset == 'ml1m' or dataset == 'douban_movie': # FORMAT: user, item, rating, timestamp
#             ls = l.strip().split("\t")
#             user_sequence.append(ls[0])
#             item_sequence.append(ls[1])
#             if start_timestamp is None:
#                 start_timestamp = float(ls[3])
#             timestamp_sequence.append(float(ls[3]) - start_timestamp) 
#         else: # FORMAT: user, item, timestamp, state label, feature list 
#             ls = l.strip().split(",")
#             user_sequence.append(ls[0])
#             item_sequence.append(ls[1])
#             if start_timestamp is None:
#                 start_timestamp = float(ls[2])
#             timestamp_sequence.append(float(ls[2]) - start_timestamp) 
#             # y_true_labels.append(int(ls[3])) # label = 1 at state change, 0 otherwise
#             feature_sequence.append(list(map(float,ls[4:])))
#     f.close()

#     user_sequence = np.array(user_sequence) 
#     item_sequence = np.array(item_sequence)
#     timestamp_sequence = np.array(timestamp_sequence)

#     print("Formating item sequence")
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

#     print("Formating user sequence")
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

#     if time_scaling:
#         print("Scaling timestamps")
#         # user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
#         # item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

#     print("*** Network loading completed ***\n\n")
#     return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
#         item2id, item_sequence_id, item_timedifference_sequence, \
#         timestamp_sequence, \
#         feature_sequence, \
#         y_true_labels]


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
        timestamp_sequence, feature_sequence, rating_sequence, sign_sequence, y_true_labels]




def create_edge_features(df, user_features, item_features):
    """유저/아이템 피처를 concat해서 edge feature 생성"""
    edge_features = []
    
    for _, row in df.iterrows():
        user_id = int(row['user'])
        item_id = int(row['item'])
        
        edge_feat = []
        
        # 유저 피처가 있는 경우에만 추가
        if user_features is not None: # and user_id in user_features:
            edge_feat.extend(user_features[user_id])
            
        # 아이템 피처가 있는 경우에만 추가  
        if item_features is not None: # and item_id in item_features:
            edge_feat.extend(item_features[item_id])
            
        # 아무 피처도 없는 경우 빈 리스트 또는 기본값
        if not edge_feat:
            edge_feat = []  # 또는 edge_feat = [0] 등 원하는 기본값
            
        edge_features.append(edge_feat)
    
    return np.array(edge_features)


'''def load_feature_with_mapping(args, item_feat_flag, user_feat_flag, user2id, item2id):
    if (item_feat_flag == True) and (user_feat_flag == True):       
        user_features = load_user_feat(args.user_feature_path)
        item_features = load_item_feat(args.item_feature_path)
        
        # 새 ID로 재매핑
        user_features = {user2id[orig_id]: feat for orig_id, feat in user_features.items() if orig_id in user2id}
        item_features = {item2id[orig_id]: feat for orig_id, feat in item_features.items() if orig_id in item2id}
        
    elif (item_feat_flag == True) and (user_feat_flag == False):
        user_features = None
        item_features = load_item_feat(args.item_feature_path)
        item_features = {item2id[orig_id]: feat for orig_id, feat in item_features.items() if orig_id in item2id}
        
    return user_features, item_features'''

def get_gpu_memory_usage():
    """GPU 메모리 사용량 반환 (MB 단위)"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # 현재 할당
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # 예약됨
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)   # 피크
        return allocated, reserved, peak
    return 0, 0, 0

def reset_gpu_memory_stats():
    """GPU 메모리 통계 리셋"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()