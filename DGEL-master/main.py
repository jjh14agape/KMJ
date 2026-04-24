# -*- coding: utf-8 -*

'''
This is training code of DGEL
The load_data and t-batch method refer to JODIE which we have already cited
If you use our code or our paper, please cite our paper:
Dynamic Graph Evolution Learning for Recommendation
published at SIGIR 2023
'''

import numpy as np
from library_data import *
import library_models as lib
from library_models import *
from copy import deepcopy
import datetime
from evaluate_all_ import eval_one_epoch
now = datetime.datetime.now()

# Initialize Parameters
parser = argparse.ArgumentParser()

# select dataset and training mode
parser.add_argument('--dataset', default="wikipedia", help='Name of the dataset')
parser.add_argument('--model', default='DGEL',type=str, help="Model name")
parser.add_argument('--gpu', default=1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding {16, 32, 64, 128}')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument('--online_test', action='store_true', help='Enable online test mode (default: false)')

# general training hyper-parameters
parser.add_argument('--weight_decay', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--learning_rate', type=float, default=1e-3, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--span_num', default=500, type=int, help='time span number')
parser.add_argument('--early_stop', type=int, default=5, help='link or link_sign or sign')
parser.add_argument('--tolerance', type=float, default=1e-3, help='tolerated marginal improvement for early stopper')

# method-related hyper-parameters
parser.add_argument('--sample_length', type=int, default=100, help='sample length {50, 100, 150, 200, 250, 300}') # reddit=150, wiki=100, lastfm=? (50, 100, 150, 200, 250, 300)
parser.add_argument('--l2u', type=float, default=1.0, help='regular coefficient of user')
parser.add_argument('--l2i', type=float, default=1.0, help='regular coefficient of item')
parser.add_argument('--bpr_coefficient', type=float, default=0.001, help='[0.001, 0.0005]')

# options
parser.add_argument('--state_change', action='store_true', help='True if training with state change of users in addition to the next interaction prediction. False otherwise. By default, set to True. MUST BE THE SAME AS THE ONE USED IN TRAINING.') 

args = parser.parse_args()

set_random_seed(args.seed)

final_embedding_dim = args.embedding_dim

args.datapath = "../signed-dataset/%s/%s.csv" % (args.dataset, args.dataset)
args.user_feature_path = "../signed-dataset/%s/user_feat.csv" % (args.dataset)
args.item_feature_path = "../signed-dataset/%s/item_feat.csv" % (args.dataset)
log_path = './log/'
best_model_root = './best_models/'
checkpoint_root = './saved_checkpoints/'

# args.datapath = "../dataset/%s/%s.csv" % (args.dataset, args.dataset)
# args.user_feature_path = "../dataset/%s/user_feat.csv" % (args.dataset)
# args.item_feature_path = "../dataset/%s/item_feat.csv" % (args.dataset)
# log_path = './log/'
# best_model_root = './best_models/'
# checkpoint_root = './saved_checkpoints/'

# args.datapath = "./DyRec/dataset/%s/%s.csv" % (args.dataset, args.dataset)
# user_feature_path = "./DyRec/dataset/%s/user_feat.csv" % (args.dataset)
# item_feature_path = "./DyRec/dataset/%s/item_feat.csv" % (args.dataset)
# log_path = 'DyRec/DGEL-master/log/'
# best_model_root = 'DyRec/DGEL-master/best_models/'
# checkpoint_root = 'DyRec/DGEL-master/saved_checkpoints/'

logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys.argv, now, log_path, checkpoint_root, best_model_root)

if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# Set your GPU here
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# # Load Data
# [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
#  item2id, item_sequence_id, item_timediffs_sequence,
#  timestamp_sequence, feature_sequence, timedifference_sequence_for_adj] = load_network(args)

 # LOAD DATA
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence,
 timestamp_sequence, feature_sequence, rating_sequence, sign_sequence, y_true, timedifference_sequence_for_adj] = load_network(args)



num_interactions = len(user_sequence_id)
num_users = len(user2id)
num_items = len(item2id) + 1  # one extra item for "none-of-these"
# num_features = len(feature_sequence[0])

# Load Feature
if args.dataset == "douban_movie":
    item_feat_flag = True
    user_feat_flag = False
    num_features = 0 
elif args.dataset == "ml1m":
    item_feat_flag = True
    user_feat_flag = True
    num_features = 0
else:
    item_feat_flag = False
    user_feat_flag = False
    num_features = len(feature_sequence[0])

if args.dataset == "douban_movie" or args.dataset == "ml1m":
    user_feature, item_feature = load_feature(args, item_feat_flag, user_feat_flag, item2id, user2id)
else: 
    user_feature, item_feature = None, None

if item_feat_flag == True:
    if num_items - 1 != len(item_feature):
        print("item feature error")
    else:
        num_features += len(item_feature[0])
if user_feat_flag == True:
    if num_users != len(user_feature):
        print("user feature error")
    else:
        num_features += len(user_feature[0])

print("***** Dataset statistics:\n  %d users\n  %d items\n  %d interactions *****\n" % (num_users, num_items, num_interactions))
   
# Set training, validation and test boundaries
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

# Set batching timespan
'''
Timespan is the frequency at which the batches are created and the DGEL model is trained.
As the data arrives in a temporal order, the interactions within a timespan are added into batches (using the T-batch algorithm).
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
Longer timespan leads to less frequent model updates.
'''

timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / args.span_num

# Initialize model and parameters
model = DGEL(args, num_features, num_users, num_items, final_embedding_dim).cuda()
MSELoss = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
early_stopper = EarlyStopMonitor(max_round=args.early_stop , tolerance=args.tolerance ) 

# Initialize embeddings
# The initial user and item embeddings are learned during training as well
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(final_embedding_dim).cuda(), dim=0))
initial_item_embedding = nn.Parameter(F.normalize(torch.rand(final_embedding_dim).cuda(), dim=0))
model.initial_user_embedding = initial_user_embedding
model.initial_item_embedding = initial_item_embedding

user_embeddings = initial_user_embedding.repeat(num_users, 1)
item_embeddings = initial_item_embedding.repeat(num_items, 1)

item_embedding_static = Variable(torch.eye(num_items).cuda())
user_embedding_static = Variable(torch.eye(num_users).cuda())



# Run training process
print("***** Training the DGEL model for %d epochs *****" % args.epochs)

user_adj = None
item_adj = None
mean_learning_time_per_epoch = []
mean_inference_time_per_epoch = []

for ep in range(args.epochs):

    # epoch_start_time = time.time()

    # Initialize embedding trajectory storage
    user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, final_embedding_dim).cuda())
    item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, final_embedding_dim).cuda())
    # predicted_ranking = torch.zeros(num_interactions, 1, dtype=torch.int, device='cuda') # mj add

    optimizer.zero_grad()
    reinitialize_tbatches()
    total_loss, loss, total_interaction_count, total_batch_count = 0, 0, 0, 0

    tbatch_start_time = None
    tbatch_to_insert = -1
    tbatch_full = False

    # Record neighbors based on interactions
    # Record timestamp of item (users) for users (items)
    user_adj = defaultdict(list)
    item_adj = defaultdict(list)
    user_timestamp_for_adj = defaultdict(list)
    item_timestamp_for_adj = defaultdict(list)

    torch.cuda.reset_peak_memory_stats() # 학습 시작 전 메모리 리셋
    torch.cuda.synchronize()
    epoch_start_time = time.time()

    for j in range(train_end_idx):
        # Read interaction
        userid = user_sequence_id[j]
        itemid = item_sequence_id[j]
        if item_feat_flag == True and user_feat_flag == True:
            feature = user_feature[userid] + item_feature[itemid]
        elif item_feat_flag == True and user_feat_flag == False:
            feature = item_feature[itemid]
        elif item_feat_flag == False and user_feat_flag == True:
            feature = user_feature[userid]
        else:
            feature = feature_sequence[j]
        user_timediff = user_timediffs_sequence[j]
        item_timediff = item_timediffs_sequence[j]
        timestamp = timestamp_sequence[j]
        timestamp_for_adj = timedifference_sequence_for_adj[j]

        user_adj[userid].append(itemid)
        item_adj[itemid].append(userid)
        user_timestamp_for_adj[userid].append(timestamp_for_adj)
        item_timestamp_for_adj[itemid].append(timestamp_for_adj)

        # to save time for number of neighbors
        length_of_user = len(user_adj[userid])
        length_of_item = len(item_adj[itemid])
        if length_of_user > args.sample_length:
            user_adj[userid] = user_adj[userid][length_of_user-args.sample_length:]
            user_timestamp_for_adj[userid] = user_timestamp_for_adj[userid][length_of_user-args.sample_length:]
        if length_of_item > args.sample_length:
            item_adj[itemid] = item_adj[itemid][length_of_item-args.sample_length:]
            item_timestamp_for_adj[itemid] = item_timestamp_for_adj[itemid][length_of_item-args.sample_length:]

        # Create T-batch and add current interaction into t-batch
        tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
        lib.tbatchid_user[userid] = tbatch_to_insert
        lib.tbatchid_item[itemid] = tbatch_to_insert

        lib.current_tbatches_user[tbatch_to_insert].append(userid)
        lib.current_tbatches_item[tbatch_to_insert].append(itemid)
        lib.current_tbatches_feature[tbatch_to_insert].append(feature)
        lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
        lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
        lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
        lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

        # lib.current_tbatches_user_history_differ[tbatch_to_insert].append([args.temperature*(timestamp_for_adj-each) for each in user_timestamp_for_adj[userid]])
        # lib.current_tbatches_item_history_differ[tbatch_to_insert].append([args.temperature*(timestamp_for_adj-each) for each in item_timestamp_for_adj[itemid]])

        lib.current_tbatches_user_history_differ[tbatch_to_insert].append([-1 * (timestamp_for_adj - each) / (timestamp_for_adj - user_timestamp_for_adj[userid][0])
                                                                           if timestamp_for_adj - user_timestamp_for_adj[userid][0] != 0 else 1 for each in user_timestamp_for_adj[userid]])
        lib.current_tbatches_item_history_differ[tbatch_to_insert].append([-1 * (timestamp_for_adj - each) / (timestamp_for_adj - item_timestamp_for_adj[itemid][0])
                                                                           if timestamp_for_adj - item_timestamp_for_adj[itemid][0] != 0 else 1 for each in item_timestamp_for_adj[itemid]])

        # current adj should not be allowed to touch future
        lib.current_tbatches_user_adj[tbatch_to_insert].append(deepcopy(user_adj[userid]))
        lib.current_tbatches_item_adj[tbatch_to_insert].append(deepcopy(item_adj[itemid]))

        if tbatch_start_time is None:
            tbatch_start_time = timestamp

        # After all interactions in the timespan are converted to t-batchs,
        # Forward pass to create embedding trajectories and calculate loss
        if timestamp - tbatch_start_time > tbatch_timespan:
            tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

            # Process t-batches
            for i in range(len(lib.current_tbatches_user)):
                total_interaction_count += len(lib.current_tbatches_interactionids[i])
                total_batch_count += len(lib.current_tbatches_user)

                # Load current t-batch
                tbatch_userids = torch.LongTensor(lib.current_tbatches_user[i]).cuda() # Recall "lib.current_tbatches_user[i]" has unique elements
                tbatch_itemids = torch.LongTensor(lib.current_tbatches_item[i]).cuda() # Recall "lib.current_tbatches_item[i]" has unique elements
                tbatch_interactionids = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                feature_tensor = Variable(torch.Tensor(lib.current_tbatches_feature[i]).cuda()) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                user_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()).unsqueeze(1)
                item_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()).unsqueeze(1)
                tbatch_itemids_previous = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()
                item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]

                tbatch_user_history_differ = lib.current_tbatches_user_history_differ[i]
                tbatch_item_history_differ = lib.current_tbatches_item_history_differ[i]

                # Project previous user embeddings to current time
                # We treat current time as previous time's future for computing convenience
                user_embedding_input = user_embeddings[tbatch_userids, :].cuda()
                item_embedding_input = item_embeddings[tbatch_itemids, :].cuda()

                # future prediction
                user_projected_embedding = model.forward(user_embedding_input, None, None, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous, :], user_embedding_static[tbatch_userids, :]], dim=1).cuda()
                predicted_item_embedding = model.predict_item_embedding(user_item_embedding)


                # mj add
                # # Distance between predicted embedding and other item embeddings
                # expanded_predicted_item_embedding = predicted_item_embedding.repeat_interleave(num_items, dim=0)
                # candidate_item_embeddings = torch.cat([item_embeddings, item_embedding_static], dim=1).detach() # 모든 아이템
                # expanded_candidate_item_embeddings = candidate_item_embeddings.repeat(len(predicted_item_embedding), 1)
                # euclidean_distances = nn.PairwiseDistance()(expanded_predicted_item_embedding, expanded_candidate_item_embeddings).squeeze(-1)
                
                # # Calculate true item rank among all the items
                # true_item_distance = []
                # for l, idx in enumerate(tbatch_itemids):
                #     start = (num_items * l)
                #     selected_value = euclidean_distances[start+idx]
                #     true_item_distance.append(selected_value)

                # # true_item_distance = euclidean_distances[itemid]
                # true_item_rank = []
                # for k, idx in enumerate(tbatch_itemids):
                #     start = (num_items * k)
                #     euclidean_distances_smaller = (euclidean_distances[start:start+num_items] < true_item_distance[k]).data.cpu().numpy()
                #     true_item_rank.append(np.sum(euclidean_distances_smaller) + 1)

                
            
                # Evolution loss for Future drifting
                # There are two parts for evolution loss:
                # 1) Future drifting from current.
                # 2) Current update from previous
                
                loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]], dim=1).detach())

                # Update dynamic sub-embeddings based on current interaction
                # Note we only update current user and item, instead of all nodes! # lib.current_tbatches_user_adj[i]: t배치 유저들의 인접 행렬
                user_adj_, user_length_mask, user_max_length = model.adj_sample(lib.current_tbatches_user_adj[i], args.sample_length)
                item_adj_, item_length_mask, item_max_length = model.adj_sample(lib.current_tbatches_item_adj[i], args.sample_length)

                user_adj_td, _, _, = model.adj_sample(tbatch_user_history_differ, args.sample_length, 'timediffer')
                item_adj_td, _, _, = model.adj_sample(tbatch_item_history_differ, args.sample_length, 'timediffer')

                user_adj_em = item_embeddings[torch.LongTensor(user_adj_).cuda(), :].cuda()
                item_adj_em = user_embeddings[torch.LongTensor(item_adj_).cuda(), :].cuda()

                # 1: inherent interaction
                user_inter_embeddings = model.interaction_aggregate(user_embedding_input, item_embedding_input, feature_tensor, user_timediffs_tensor, 'user')
                item_inter_embeddings = model.interaction_aggregate(item_embedding_input, user_embedding_input, feature_tensor, item_timediffs_tensor, 'item')

                # 2: time-decay neighbor GCN
                user_adj_embedding = model.neighbor_aggregate(user_embedding_input, user_adj_em, torch.LongTensor(user_length_mask).cuda(), user_max_length, torch.FloatTensor(user_adj_td).cuda(), 'user')
                item_adj_embedding = model.neighbor_aggregate(item_embedding_input, item_adj_em, torch.LongTensor(item_length_mask).cuda(), item_max_length, torch.FloatTensor(item_adj_td).cuda(), 'item')

                # 3: symbiotic local learning
                user_ext_embeddings = model.excitement_aggregate(user_adj_em, torch.LongTensor(user_length_mask).cuda(), user_max_length)
                item_ext_embeddings = model.excitement_aggregate(item_adj_em, torch.LongTensor(item_length_mask).cuda(), item_max_length)
                user_local_embedding, item_local_embedding = model.local_aggregate(user_embedding_input, item_embedding_input, user_ext_embeddings, item_ext_embeddings)

                # forward with re-scaling network
                user_embedding_output = model.forward(user_embedding_input, user_inter_embeddings, user_local_embedding,
                                                      timediffs=user_timediffs_tensor, features=feature_tensor,
                                                      adj_embeddings=user_adj_embedding, select='user_update')
                item_embedding_output = model.forward(item_embedding_input, item_inter_embeddings, item_local_embedding,
                                                      timediffs=item_timediffs_tensor, features=feature_tensor,
                                                      adj_embeddings=item_adj_embedding, select='item_update')

                # Save embeddings
                item_embeddings[tbatch_itemids, :] = item_embedding_output
                user_embeddings[tbatch_userids, :] = user_embedding_output
                user_embeddings_timeseries[tbatch_interactionids, :] = user_embedding_output
                item_embeddings_timeseries[tbatch_interactionids, :] = item_embedding_output
                
                # mj add
                # true_item_rank_tensor = torch.tensor(true_item_rank, dtype=torch.int, device=predicted_ranking.device)
                # true_item_rank_tensor = true_item_rank_tensor.unsqueeze(1)
                # predicted_ranking[tbatch_interactionids, :] = true_item_rank_tensor
                
                # Evolution loss for current updating from previous
                loss += args.l2i*MSELoss(item_embedding_output.cuda(), item_embedding_input.cuda().detach())
                loss += args.l2u*MSELoss(user_embedding_output.cuda(), user_embedding_input.cuda().detach())

                # sample negative item for BPR-loss
                neg_items = model.sample_for_BPR(lib.current_tbatches_user[i], num_items-1, lib.current_tbatches_user_adj[i], bpr_avoid=None)  # num_items = len(item2Id) + 1
                neg_item_embeddings = item_embeddings[torch.LongTensor(neg_items).cuda(), :]

                # BPR-loss for current t-batch
                # The bpr loss is extremely bigger like 0.79 but each MSE loss is so small like 0.0008
                # So the bpr_coefficient being 0.001 could balance the two task loss
                bpr_loss = model.bpr_loss(user_embedding_output, item_embedding_output, neg_item_embeddings.detach())
                loss += args.bpr_coefficient*bpr_loss

            # Back-propagate of t-batches
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Reset for next t-batch
            loss = 0
            item_embeddings.detach_()  # Detachment is needed to prevent double propagation of gradient
            user_embeddings.detach_()
            item_embeddings_timeseries.detach_()
            user_embeddings_timeseries.detach_()
            # mj add
            # predicted_ranking.detach_()
            

            # Re-Initialization
            reinitialize_tbatches()
            tbatch_to_insert = -1

    
    # learning_time_per_epoch = time.time()-epoch_start_time
    # mean_learning_time_per_epoch.append(learning_time_per_epoch)
    # print("\n\n{} epoch took {} seconds".format(ep, learning_time_per_epoch))
    torch.cuda.synchronize()
    learning_time_per_epoch = time.time()-epoch_start_time

    # 학습 피크 메모리 측정
    train_peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    train_current_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    if ep > 0:  
        mean_learning_time_per_epoch.append(learning_time_per_epoch)

    val_mrr, val_rec1, val_rec5, val_rec10, val_rec20, val_pre1, val_pre5, val_pre10, val_pre20, val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20, inference_time_per_epoch, val_peak_memory_mb = eval_one_epoch(args, model, optimizer, MSELoss, user_sequence_id, item_sequence_id, feature_sequence, item_feat_flag, user_feat_flag, item_feature, user_feature, 
                   user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, timedifference_sequence_for_adj,
                   user_embeddings, item_embeddings, user_embedding_static, item_embedding_static, user_embeddings_timeseries, item_embeddings_timeseries,
                   user_adj, item_adj, user_timestamp_for_adj, item_timestamp_for_adj, validation_start_idx, test_start_idx)
    
    mean_inference_time_per_epoch.append(inference_time_per_epoch)
    # End of epoch
    logger.info('epoch: {}'.format(ep))
    logger.info('Total loss in this epoch: {}'.format(total_loss))
    # 같은 메트릭끼리 한 줄에 탭으로 구분하여 출력
    logger.info('val MRR: {:.4f}'.format(val_mrr))
    logger.info('val REC:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_rec1, val_rec5, val_rec10, val_rec20))
    logger.info('val PRE:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_pre1, val_pre5, val_pre10, val_pre20))
    logger.info('val NDCG:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20))
    logger.info('Trainig time: {} sec'.format(learning_time_per_epoch))
    logger.info('Inference time: {} sec'.format(inference_time_per_epoch))
    logger.info('Training peak memory: {:.2f} MiB (current: {:.2f} MiB)'.format(train_peak_memory_mb, train_current_memory_mb))  # 추가
    logger.info('Inference peak memory: {:.2f} MiB'.format(val_peak_memory_mb))  # 추가

    # End of epoch
    if early_stopper.early_stop_check(val_mrr):
        logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_checkpoint_path = get_checkpoint_path(early_stopper.best_epoch)
        # model.load_state_dict(torch.load(best_checkpoint_path, weights_only=True))
        model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_adj, item_adj, user_timestamp_for_adj, item_timestamp_for_adj, \
            user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, best_checkpoint_path)
        set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx)

        # Load embeddings
        item_embeddings = item_embeddings_dystat[:, :final_embedding_dim]
        item_embeddings = item_embeddings.clone()
        item_embedding_static = item_embeddings_dystat[:, final_embedding_dim:]
        item_embedding_static = item_embedding_static.clone()

        user_embeddings = user_embeddings_dystat[:, :final_embedding_dim]
        user_embeddings = user_embeddings.clone()
        user_embedding_static = user_embeddings_dystat[:, final_embedding_dim:]
        user_embedding_static = user_embedding_static.clone()

        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        model.eval()
        break
    else:
        # torch.save(model.state_dict(), get_checkpoint_path(ep))
        item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
        user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)

        save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx,
               user_adj, item_adj, user_timestamp_for_adj, item_timestamp_for_adj, user_embeddings_timeseries, item_embeddings_timeseries, get_checkpoint_path(ep))

        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        item_embeddings = initial_item_embedding.repeat(num_items, 1)



# Save final model of final epoch
# Save final model of final epoch
logger.info("\n***** Training complete. Testing final model. *****\n")
test_mrr, test_rec1, test_rec5, test_rec10, test_rec20, test_pre1, test_pre5, test_pre10, test_pre20, test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20, inference_time, test_peak_memory_mb = eval_one_epoch(args, model, optimizer, MSELoss, user_sequence_id, item_sequence_id, feature_sequence, item_feat_flag, user_feat_flag, item_feature, user_feature, 
                   user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, timedifference_sequence_for_adj,
                   user_embeddings, item_embeddings, user_embedding_static, item_embedding_static, user_embeddings_timeseries, item_embeddings_timeseries,
                   user_adj, item_adj, user_timestamp_for_adj, item_timestamp_for_adj, test_start_idx, test_end_idx)
# logger.info("Remove Loss")
logger.info('Best epoch: {}'.format(early_stopper.best_epoch))
logger.info('val MRR:\t{:.4f}'.format(test_mrr))
logger.info('val REC:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_rec1, test_rec5, test_rec10, test_rec20))
logger.info('val PRE:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_pre1, test_pre5, test_pre10, test_pre20))
logger.info('val NDCG:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20))
logger.info('Average training time per epoch: {}'.format(np.mean(mean_learning_time_per_epoch)))
logger.info('Average inference time per epoch: {}'.format(np.mean(mean_inference_time_per_epoch)))
logger.info('Inference time (testing): {} sec'.format(inference_time))
logger.info('Inference peak memory: {:.2f} MiB'.format(test_peak_memory_mb))  # 추가


torch.save(model.state_dict(), best_model_path)