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
parser.add_argument('--weight_decay', type=float, default=1e-3, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
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
parser.add_argument('--sample_length', type=int, default=80, help='sample length for second order relationship') # reddit=60, wiki=80, lastfm=20 # [20, 40, 60, 80, 100, 120]
parser.add_argument('--l2u', type=float, default=1.0, help='regular coefficient of user')
parser.add_argument('--l2i', type=float, default=1.0, help='regular coefficient of item')

parser.add_argument('--best_epoch', type=int, default=15, help='random seed for all randomized algorithms')
parser.add_argument('--runtime_id', type=str, default='2025-8-16-16-20-42-wikipedia', help='random seed for all randomized algorithms')

# options
parser.add_argument('--state_change', action='store_true', help='True if training with state change of users in addition to the next interaction prediction. False otherwise. By default, set to True. MUST BE THE SAME AS THE ONE USED IN TRAINING.') 

args = parser.parse_args()
print(args)
set_random_seed(args.seed)

args.datapath = "../dataset/%s/%s.csv" % (args.dataset, args.dataset)
args.user_feature_path = "../dataset/%s/user_feat.csv" % (args.dataset)
args.item_feature_path = "../dataset/%s/item_feat.csv" % (args.dataset)
log_path = './log/'
best_model_root = './best_models/'
checkpoint_root = './saved_checkpoints/'

logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys.argv, args.runtime_id, log_path, checkpoint_root, best_model_root)


# LOAD DATA
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence, 
 timestamp_sequence, feature_sequence, y_true] = load_network(args)

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
    user_feature, item_feature = load_feature(args, item_feat_flag, user_feat_flag)
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


# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

MSELoss = nn.MSELoss()
model = DGCF(args, num_features, num_users, num_items).cuda()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

best_checkpoint_path = get_checkpoint_path(args.best_epoch)
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

logger.info(f'Loaded the best model at epoch {args.best_epoch} for inference')
model.eval()
    
# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
# print("\n\n*** Training complete. Saving final model. ***\n\n")
# print("\n\n*** Average time per epoch {} seconds. ***\n\n".format(np.mean(mean_learning_time_per_epoch)))
# save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_adj, item_adj, user_embeddings_timeseries, item_embeddings_timeseries)


# Save final model of final epoch
logger.info("\n***** Training complete. Testing final model. *****\n")
test_mrr, test_rec1, test_rec5, test_rec10, test_rec20, test_pre1, test_pre5, test_pre10, test_pre20, test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20, inference_time = eval_one_epoch(args, model, optimizer, MSELoss, user_sequence_id, item_sequence_id, feature_sequence, item_feat_flag, user_feat_flag, item_feature, user_feature, 
                   user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, 
                   user_embeddings, item_embeddings, user_embedding_static, item_embedding_static, user_embeddings_timeseries, item_embeddings_timeseries,
                   user_adj, item_adj, test_start_idx, test_end_idx, num_users)
logger.info('Best epoch: {}'.format(args.best_epoch))
logger.info('test MRR:\t{:.4f}'.format(test_mrr))
logger.info('test REC:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_rec1, test_rec5, test_rec10, test_rec20))
logger.info('test PRE:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_pre1, test_pre5, test_pre10, test_pre20))
logger.info('test NDCG:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20))
# logger.info('Average training time per epoch: {}'.format(np.mean(mean_learning_time_per_epoch)))
# logger.info('Average inference time per epoch: {}'.format(np.mean(mean_inference_time_per_epoch)))
logger.info('Inference time (testing): {} sec'.format(inference_time))

torch.save(model.state_dict(), best_model_path)