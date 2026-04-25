# -*- coding: utf-8 -*
'''
This code trains the DGCF model for the given dataset.
The task is: interaction prediction.

How to run: 
$ python DGCF.py --network reddit --model DGCF --epochs 50

Reference Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019.
'''
import time
from library_data import *
import library_models as lib
from library_models import *
import datetime
# from IPython import embed
from evaluate_all_ import eval_one_epoch
now = datetime.datetime.now()
# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()

# select dataset and training mode
parser.add_argument('--dataset', default="wikipedia", help='Name of the network/dataset') # wikipedia, lastfm, mooc, reddit, douban_movie, ml1m
parser.add_argument('--model', default="DGCF", help='Model name to save output in file')
parser.add_argument('--gpu', default=1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding { 128 }')
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
# store_true = 인자 입력하면 ture, 입력 안하면 자동 false
parser.add_argument('--adj', default=True, type=bool, help='The second order relationship') # 2hop agg 
parser.add_argument('--no_zero', action='store_true', help='The zero order relationship (default: false)') # interaction aff
parser.add_argument('--no_first', action='store_true', help='The first order relationship (default: false)') # interaction agg 
parser.add_argument('--method', default="gat", help='The way of aggregate adj')
parser.add_argument('--sample_length', type=int, default=None, help='sample length for second order relationship') # reddit=60, wiki=80, lastfm=20 # [20, 40, 60, 80, 100, 120]
parser.add_argument('--l2u', type=float, default=1.0, help='regular coefficient of user')
parser.add_argument('--l2i', type=float, default=1.0, help='regular coefficient of item')

# options
parser.add_argument('--state_change', action='store_true', help='True if training with state change of users in addition to the next interaction prediction. False otherwise. By default, set to True. MUST BE THE SAME AS THE ONE USED IN TRAINING.') 

args = parser.parse_args()
print(args)
set_random_seed(args.seed)

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
# log_path = 'DyRec/jodie-master/log/'
# best_model_root = 'DyRec/jodie-master/best_models/'
# checkpoint_root = 'DyRec/jodie-master/saved_checkpoints/'

logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys.argv, now, log_path, checkpoint_root, best_model_root)

if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# SET GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# # LOAD DATA
# [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
#  item2id, item_sequence_id, item_timediffs_sequence, 
#  timestamp_sequence, feature_sequence, y_true] = load_network(args)

# LOAD DATA Sign-aware add
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence,
 timestamp_sequence, feature_sequence, rating_sequence, sign_sequence, y_true] = load_network(args)

num_interactions = len(user_sequence_id)
num_users = len(user2id) 
num_items = len(item2id) + 1 # one extra item for "none-of-these"

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

# num_features = len(feature_sequence[0])
true_labels_ratio = len(y_true)/(1.0+sum(y_true)) # +1 in denominator in case there are no state change labels, which will throw an error.
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

# SET BATCHING TIMESPAN
'''
Timespan is the frequency at which the batches are created and the DGCF model is trained.
As the data arrives in a temporal order, the interactions within a timespan are added into batches (using the T-batch algorithm).
The batches are then used to train DGCF.
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
Longer timespan leads to less frequent model updates.
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]   #总的时间间隔
tbatch_timespan = timespan / args.span_num                           #

# INITIALIZE MODEL AND PARAMETERS
model = DGCF(args, num_features, num_users, num_items).cuda()
weight = torch.Tensor([1, true_labels_ratio]).cuda()
crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()
early_stopper = EarlyStopMonitor(max_round=args.early_stop , tolerance=args.tolerance ) 

# INITIALIZE EMBEDDING
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0)) # the initial user and item embeddings are learned during training as well
initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
model.initial_user_embedding = initial_user_embedding
model.initial_item_embedding = initial_item_embedding

user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding
item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding
item_embedding_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings
user_embedding_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings

# INITIALIZE MODEL
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# RUN THE DGCF MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, DGCF USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
'''
print("*** Training the DGCF model for %d epochs ***" % args.epochs)
#with trange(args.epochs) as progress_bar1:
user_adj = None
item_adj = None
mean_learning_time_per_epoch = []
mean_inference_time_per_epoch = []

for ep in range(args.epochs):
    #progress_bar1.set_description('Epoch %d of %d' % (ep, args.epochs))

    # epoch_start_time = time.time()
    # INITIALIZE EMBEDDING TRAJECTORY STORAGE
    user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
    item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())

    optimizer.zero_grad()
    reinitialize_tbatches()
    total_loss, loss, total_interaction_count = 0, 0, 0

    tbatch_start_time = None
    tbatch_to_insert = -1
    tbatch_full = False

    torch.cuda.reset_peak_memory_stats() # 학습 시작 전 메모리 리셋
    torch.cuda.synchronize()
    epoch_start_time = time.time()

    if args.adj or args.no_zero or args.no_first:
        if not args.sample_length:
            user_adj = defaultdict(set)  # 每个user的邻居
            item_adj = defaultdict(set)  # 每个item的邻居
        else:
            user_adj = defaultdict(list)  # 每个user的邻居
            item_adj = defaultdict(list)  # 每个item的邻居

    # TRAIN TILL THE END OF TRAINING INTERACTION IDX
    #with trange(train_end_idx) as progress_bar2:
    for j in range(train_end_idx):
        #progress_bar2.set_description('Processed %dth interactions' % j)

        # READ INTERACTION J
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
        if args.adj or args.no_zero or args.no_first:
            # 计算user和item的邻居：
            if not args.sample_length:
                user_adj[userid].add(itemid)  # 实时更新user和item的邻居 user_adj is dic, key is user_id ,value is item_id
                item_adj[itemid].add(userid)
            else:
                user_adj[userid].append(itemid)  # 实时更新user和item的邻居 user_adj is dic, key is user_id ,value is item_id
                item_adj[itemid].append(userid)
        
        # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
        tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1   #tbatch_user:user_id所在的batch  tbatch_item:item_id所在的batch
        lib.tbatchid_user[userid] = tbatch_to_insert                                       #为了保证同一个batch没有相同的item和user
        lib.tbatchid_item[itemid] = tbatch_to_insert

        lib.current_tbatches_user[tbatch_to_insert].append(userid)     #每个batch中的user/item/feature/interactions/timediffs....都用list存
        lib.current_tbatches_item[tbatch_to_insert].append(itemid)
        lib.current_tbatches_feature[tbatch_to_insert].append(feature)
        lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
        lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
        lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
        lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

        if args.adj or args.no_zero or args.no_first:
            # batch中每个user和item的邻居
            lib.current_tbatches_user_adj[tbatch_to_insert].append(user_adj[userid])  # items
            lib.current_tbatches_item_adj[tbatch_to_insert].append(item_adj[itemid])  # user

        timestamp = timestamp_sequence[j]
        if tbatch_start_time is None:
            tbatch_start_time = timestamp

        # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES,
        # FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
        # after all interactions in the timespan are converted to t-batchs,
        # forward pass to crate embedding trajectories and calculate prediction loss
        if timestamp - tbatch_start_time > tbatch_timespan:
            tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

            # ITERATE OVER ALL T-BATCHES
            #with trange(len(lib.current_tbatches_user)) as progress_bar3:   #len(lib.current_tbatches_user):当前batch的数量
            for i in range(len(lib.current_tbatches_user)):
                #progress_bar3.set_description('Processed %d of %d T-batches ' % (i, len(lib.current_tbatches_user)))

                total_interaction_count += len(lib.current_tbatches_interactionids[i])

                # LOAD THE CURRENT TBATCH
                tbatch_userids = torch.LongTensor(lib.current_tbatches_user[i]).cuda() # Recall "lib.current_tbatches_user[i]" has unique elements
                tbatch_itemids = torch.LongTensor(lib.current_tbatches_item[i]).cuda() # Recall "lib.current_tbatches_item[i]" has unique elements
                tbatch_interactionids = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                feature_tensor = Variable(torch.Tensor(lib.current_tbatches_feature[i]).cuda()) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                user_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()).unsqueeze(1)
                item_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()).unsqueeze(1)
                tbatch_itemids_previous = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()
                item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]

                # PROJECT USER EMBEDDING TO CURRENT TIME
                user_embedding_input = user_embeddings[tbatch_userids,:]
                user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous,:], user_embedding_static[tbatch_userids,:]], dim=1)

                # PREDICT NEXT ITEM EMBEDDING
                predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                # CALCULATE PREDICTION LOSS
                item_embedding_input = item_embeddings[tbatch_itemids,:]
                loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]], dim=1).detach())

                # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                if args.adj or args.no_zero or args.no_first:
                    if not args.sample_length:
                        user_adj_, user_length_mask, user_max_length = adj_pad(lib.current_tbatches_user_adj[i])
                        item_adj_, item_length_mask, item_max_length = adj_pad(lib.current_tbatches_item_adj[i])
                    else:
                        user_adj_, user_length_mask, user_max_length = adj_sample(lib.current_tbatches_user_adj[i],
                                                                               args.sample_length)
                        item_adj_, item_length_mask, item_max_length = adj_sample(lib.current_tbatches_item_adj[i],
                                                                               args.sample_length)
                    user_adj_em = item_embeddings[torch.LongTensor(user_adj_).cuda(), :] # item embedding aggregation 한 것
                    item_adj_em = user_embeddings[torch.LongTensor(item_adj_).cuda(), :]
                    
                    
                    if model.method == 'mean':
                        # user_adj_embeddin4c3a814e-d4ea-4c6e-87a8-5f1d38933246g = model.aggregate(item_embeddings, lib.current_tbatches_user_adj[i], select='user_update')
                        # item_adj_embedding = model.aggregate(user_embeddings, lib.current_tbatches_item_adj[i], select='item_update')
                        user_adj_embedding = model.aggregate_mean(user_adj_em, torch.LongTensor(user_length_mask),
                                                                  user_max_length, user_embedding_input,
                                                                  select='user_update')
                        item_adj_embedding = model.aggregate_mean(item_adj_em, torch.LongTensor(item_length_mask),
                                                                  item_max_length, item_embedding_input,
                                                                  select='item_update')
                    elif model.method == 'attention':
                        user_adj_embedding = model.aggregate_attention(user_adj_em, torch.LongTensor(user_length_mask),
                                                                       user_max_length, user_embedding_input,
                                                                       select='user_update')
                        item_adj_embedding = model.aggregate_attention(item_adj_em, torch.LongTensor(item_length_mask),
                                                                       item_max_length, item_embedding_input,
                                                                       select='item_update')
                    elif model.method == 'gat':
                        user_adj_embedding = model.aggregate_gat(user_adj_em, torch.LongTensor(user_length_mask),
                                                                       user_max_length, user_embedding_input,
                                                                       select='user_update')
                        item_adj_embedding = model.aggregate_gat(item_adj_em, torch.LongTensor(item_length_mask),
                                                                       item_max_length, item_embedding_input,
                                                                       select='item_update')
                    elif model.method == 'lstm':
                        user_adj_embedding = model.aggregate_lstm(user_adj_em, torch.LongTensor(user_length_mask),
                                                                 user_max_length, user_embedding_input,
                                                                 select='user_update')
                        item_adj_embedding = model.aggregate_lstm(item_adj_em, torch.LongTensor(item_length_mask),
                                                                 item_max_length, item_embedding_input,
                                                                 select='item_update')
                else:
                    user_adj_embedding = None
                    item_adj_embedding = None

                user_embedding_output = model.forward(user_embedding_input, item_embedding_input,
                                                      timediffs=user_timediffs_tensor, features=feature_tensor,
                                                      adj_embeddings=user_adj_embedding, select='user_update')
                item_embedding_output = model.forward(user_embedding_input, item_embedding_input,
                                                      timediffs=item_timediffs_tensor, features=feature_tensor,
                                                      adj_embeddings=item_adj_embedding, select='item_update')
                # user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                # item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

                item_embeddings[tbatch_itemids, :] = item_embedding_output
                user_embeddings[tbatch_userids, :] = user_embedding_output

                #user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                #item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output

                # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                loss += args.l2i*MSELoss(item_embedding_output, item_embedding_input.detach())
                loss += args.l2u*MSELoss(user_embedding_output, user_embedding_input.detach())

                # CALCULATE STATE CHANGE LOSS
                # if args.state_change:
                #     loss += calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_timeseries, y_true, crossEntropyLoss)


            # BACKPROPAGATE ERROR AFTER END OF T-BATCH
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # RESET LOSS FOR NEXT T-BATCH
            loss = 0
            item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
            user_embeddings.detach_()
            item_embeddings_timeseries.detach_()
            user_embeddings_timeseries.detach_()

            # REINITIALIZE
            reinitialize_tbatches()
            tbatch_to_insert = -1
    
    torch.cuda.synchronize()
    learning_time_per_epoch = time.time()-epoch_start_time

    # 학습 피크 메모리 측정
    train_peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    train_current_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    if ep > 0:  
        # learning_time_per_epoch = time.time()-epoch_start_time
        mean_learning_time_per_epoch.append(learning_time_per_epoch)
    
    val_mrr, val_rec1, val_rec5, val_rec10, val_rec20, val_pre1, val_pre5, val_pre10, val_pre20, val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20, inference_time_per_epoch, val_peak_memory_mb = eval_one_epoch(args, model, optimizer, MSELoss, user_sequence_id, item_sequence_id, feature_sequence, item_feat_flag, user_feat_flag, item_feature, user_feature, 
                   user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, 
                   user_embeddings, item_embeddings, user_embedding_static, item_embedding_static, user_embeddings_timeseries, item_embeddings_timeseries,
                   user_adj, item_adj, validation_start_idx, test_start_idx, num_users)
    
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
    
    # print("\n\n{} epoch took {} seconds".format(ep, learning_time_per_epoch))
    # # END OF ONE EPOCH
    # print("\nTotal loss in this epoch = %f" % (total_loss))
    # item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
    # user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
    # # SAVE CURRENT MODEL TO DISK TO BE USED IN EVALUATION.
    # save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx,
    #            user_adj, item_adj, user_embeddings_timeseries, item_embeddings_timeseries)

    if early_stopper.early_stop_check(val_mrr):
        logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_checkpoint_path = get_checkpoint_path(early_stopper.best_epoch)
        # model.load_state_dict(torch.load(best_checkpoint_path, weights_only=True))
        model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_adj, item_adj, \
        user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, best_checkpoint_path)
        set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx)

        # Load embeddings
        item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
        item_embeddings = item_embeddings.clone()
        item_embedding_static = item_embeddings_dystat[:, args.embedding_dim:]
        item_embedding_static = item_embedding_static.clone()

        user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
        user_embeddings = user_embeddings.clone()
        user_embedding_static = user_embeddings_dystat[:, args.embedding_dim:]
        user_embedding_static = user_embedding_static.clone()

        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        model.eval()
        break
    else:
        # torch.save(model.state_dict(), get_checkpoint_path(ep))
        item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
        user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)

        save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx,
               user_adj, item_adj, user_embeddings_timeseries, item_embeddings_timeseries, get_checkpoint_path(ep))

        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        item_embeddings = initial_item_embedding.repeat(num_items, 1)

# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
# print("\n\n*** Training complete. Saving final model. ***\n\n")
# print("\n\n*** Average time per epoch {} seconds. ***\n\n".format(np.mean(mean_learning_time_per_epoch)))
# save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_adj, item_adj, user_embeddings_timeseries, item_embeddings_timeseries)


# Save final model of final epoch
logger.info("\n***** Training complete. Testing final model. *****\n")
test_mrr, test_rec1, test_rec5, test_rec10, test_rec20, test_pre1, test_pre5, test_pre10, test_pre20, test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20, inference_time, test_peak_memory_mb = eval_one_epoch(args, model, optimizer, MSELoss, user_sequence_id, item_sequence_id, feature_sequence, item_feat_flag, user_feat_flag, item_feature, user_feature, 
                   user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, 
                   user_embeddings, item_embeddings, user_embedding_static, item_embedding_static, user_embeddings_timeseries, item_embeddings_timeseries,
                   user_adj, item_adj, test_start_idx, test_end_idx, num_users)
logger.info('Best epoch: {}'.format(early_stopper.best_epoch))
logger.info('test MRR:\t{:.4f}'.format(test_mrr))
logger.info('test REC:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_rec1, test_rec5, test_rec10, test_rec20))
logger.info('test PRE:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_pre1, test_pre5, test_pre10, test_pre20))
logger.info('test NDCG:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20))
logger.info('Average training time per epoch: {}'.format(np.mean(mean_learning_time_per_epoch)))
logger.info('Average inference time per epoch: {}'.format(np.mean(mean_inference_time_per_epoch)))
logger.info('Inference time (testing): {} sec'.format(inference_time))
logger.info('Inference peak memory: {:.2f} MiB'.format(test_peak_memory_mb))  # 추가

torch.save(model.state_dict(), best_model_path)