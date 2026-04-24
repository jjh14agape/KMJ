# -*- coding: utf-8 -*

'''
This is evaluating code of DGEL
If you use our code or our paper, please cite our paper:
Dynamic Graph Evolution Learning for Recommendation
published at SIGIR 2023
'''

from library_data import *
from library_models import *
import datetime

# Initialize Parameterss
parser = argparse.ArgumentParser()

# select dataset and training mode
parser.add_argument('--dataset', default="wikipedia", help='Name of the dataset')
parser.add_argument('--model', default='DGEL',type=str, help="Model name")
parser.add_argument('--gpu', default=1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument('--online_test', action='store_true', help='Enable online test mode')

# general training hyper-parameters
parser.add_argument('--weight_decay', type=float, default=1e-2, help='l2 penalty')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--span_num', default=500, type=int, help='time span number')

# method-related hyper-parameters
parser.add_argument('--sample_length', type=int, default=100, help='sample length {50, 100, 150, 200, 250, 300}') # reddit=150, wiki=100, lastfm=?
parser.add_argument('--l2u', type=float, default=1.0, help='regular coefficient of user')
parser.add_argument('--l2i', type=float, default=1.0, help='regular coefficient of item')
parser.add_argument('--bpr_coefficient', type=float, default=0.001, help='[0.001, 0.0005]')

# options
parser.add_argument('--state_change', action='store_true', help='True if training with state change of users in addition to the next interaction prediction. False otherwise. By default, set to True. MUST BE THE SAME AS THE ONE USED IN TRAINING.') 

args = parser.parse_args()
print(args) 
if args.online_test:
    print("Online test mode enabled.")
    output_fname = "results/prediction_on_test_%s_size%d_lr%s_wd%s_seed%d_sample_length%d_online_test.txt" % (args.dataset, args.embedding_dim, str(args.learning_rate), str(args.weight_decay), args.seed, args.sample_length)
else:
    print("Offline test mode.")
    output_fname = "results/prediction_on_test_%s_size%d_lr%s_wd%s_seed%d_sample_length%d.txt" % (args.dataset, args.embedding_dim, str(args.learning_rate), str(args.weight_decay), args.seed, args.sample_length)

set_random_seed(args.seed)

final_embedding_dim = args.embedding_dim

args.datapath = "../dataset/%s/%s.csv" % (args.dataset, args.dataset)
args.user_feature_path = "../dataset/%s/user_feat.csv" % (args.dataset)
args.item_feature_path = "../dataset/%s/item_feat.csv" % (args.dataset)
# args.datapath = "./DyRec/dataset/%s/%s.csv" % (args.dataset, args.dataset)
# user_feature_path = "./DyRec/dataset/%s/user_feat.csv" % (args.dataset)
# item_feature_path = "./DyRec/dataset/%s/item_feat.csv" % (args.dataset)

if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# Set your GPU here
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)



# Load Data
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence,
 timestamp_sequence, feature_sequence, timedifference_sequence_for_adj] = load_network(args)
num_interactions = len(user_sequence_id)
# num_features = len(feature_sequence[0])
num_users = len(user2id)
num_items = len(item2id) + 1

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
test_start_idx = int(num_interactions * (args.train_proportion + 0.1))
test_end_idx = int(num_interactions * (args.train_proportion + 0.2))

# Set batching timespan
'''
Timespan indicates how frequently the model is run and updated. 
All interactions in one timespan are processed simultaneously. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
At the end of each timespan, the model is updated as well. So, longer timespan means less frequent model updates. 
'''

timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / args.span_num

# Initialize model and parameters
model = DGEL(args, num_features, num_users, num_items, final_embedding_dim).cuda()
MSELoss = nn.MSELoss()

optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

mean_inference_time_per_epoch = []
for epp in range(12, args.epochs):
    # Check if the output of the epoch is already processed. If so, move on.
    if os.path.exists(output_fname):
        f = open(output_fname, "r")
        search_string = 'Test performance of epoch %d' % epp
        for l in f:
            l = l.strip()
            if search_string in l:
                print("Output file already has results of epoch %d" % epp)
                #sys.exit(0)
                continue
        f.close()

    # Load Trained Model
    model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_adj, item_adj, \
    user_timestamp_for_adj, item_timestamp_for_adj, user_embeddings_timeseries, \
    item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, epp)

    if train_end_idx != train_end_idx_training:
        sys.exit('Training proportion during training and testing are different. Aborting.')

    # Set the user and item embeddings to state at the end of the training period
    set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx)

    # Load embeddings
    item_embeddings = item_embeddings_dystat[:, :final_embedding_dim]
    item_embeddings = item_embeddings.clone()
    item_embeddings_static = item_embeddings_dystat[:, final_embedding_dim:]
    item_embeddings_static = item_embeddings_static.clone()

    user_embeddings = user_embeddings_dystat[:, :final_embedding_dim]
    user_embeddings = user_embeddings.clone()
    user_embeddings_static = user_embeddings_dystat[:, final_embedding_dim:]
    user_embeddings_static = user_embeddings_static.clone()

    # Performance record
    validation_ranks = []
    test_ranks = []

    ''' 
    Here we use the trained model to make predictions for the validation and testing interactions.
    The model does a forward pass from the start of validation till the end of testing.
    For each interaction, the trained model is used to predict the embedding of the item it will interact with. 
    This is used to calculate the rank of the true item the user actually interacts with.
    
    After this prediction, the errors in the prediction are used to calculate the loss and update the model parameters. 
    This simulates the real-time feedback about the predictions that the model gets when deployed in-the-wild. 
    '''

    tbatch_start_time = None
    loss = 0
    print("***** Making interaction predictions by forward pass (no t-batching) *****")
    epoch_start_time = time.time()

    for j in range(train_end_idx, test_end_idx):

        # Load test j
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

        if not tbatch_start_time:
            tbatch_start_time = timestamp
        itemid_previous = user_previous_itemid_sequence[j]

        # Load user and item embeddings
        user_embedding_input = user_embeddings[torch.cuda.LongTensor([userid])].cuda()
        user_embedding_static_input = user_embeddings_static[torch.cuda.LongTensor([userid])].cuda()
        item_embedding_input = item_embeddings[torch.cuda.LongTensor([itemid])].cuda()
        item_embedding_static_input = item_embeddings_static[torch.cuda.LongTensor([itemid])].cuda()
        feature_tensor = Variable(torch.Tensor(feature).cuda()).unsqueeze(0)
        user_timediffs_tensor = Variable(torch.Tensor([user_timediff]).cuda()).unsqueeze(0)
        item_timediffs_tensor = Variable(torch.Tensor([item_timediff]).cuda()).unsqueeze(0)
        item_embedding_previous = item_embeddings[torch.cuda.LongTensor([itemid_previous])]

        # Project previous user embeddings to current time
        user_projected_embedding = model.forward(user_embedding_input, None, None, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
        user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[torch.cuda.LongTensor([itemid_previous])], user_embedding_static_input], dim=1)

        # future prediction
        predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

        # CALCULATE PREDICTION LOSS
        if args.online_test:
            loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static_input], dim=1).detach())

        # Distance between predicted embedding and other item embeddings
        euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1), torch.cat([item_embeddings, item_embeddings_static], dim=1).detach()).squeeze(-1)

        # Calculate true item rank among all the items
        true_item_distance = euclidean_distances[itemid]
        euclidean_distances_smaller = (euclidean_distances < true_item_distance).data.cpu().numpy()
        true_item_rank = np.sum(euclidean_distances_smaller) + 1

        if j < test_start_idx:
            validation_ranks.append(true_item_rank)
        else:
            test_ranks.append(true_item_rank)

        # hold
        user_inter_embeddings, item_inter_embeddings = None, None
        user_adj_embedding, item_adj_embedding = None, None
        user_local_embedding, item_local_embedding = None, None
        user_ext_embeddings, item_ext_embeddings = None, None

        # Update dynamic sub-embeddings based on current interaction
        # Note we only update current user and item, instead of all nodes!
        if len(user_adj[userid]) == 0:
            user_adj_embedding = Variable(torch.zeros(1, model.embedding_dim).cuda()).cuda()
            user_ext_embeddings = Variable(torch.zeros(1, model.final_embedding_dim).cuda()).cuda()
        else:
            user_adj_, user_length_mask, user_max_length = model.adj_sample([user_adj[userid]], model.sample_length)
            user_adj_em = item_embeddings[torch.LongTensor(user_adj_).cuda(), :].cuda()

            user_time = [[-1 * (timestamp_for_adj - each) / (timestamp_for_adj - user_timestamp_for_adj[userid][0]) if timestamp_for_adj - user_timestamp_for_adj[userid][0] != 0 else 1 for each in user_timestamp_for_adj[userid]]]
            user_adj_td, _, _, = model.adj_sample(user_time, model.sample_length, 'timediffer')

            # time_decay neighbor GCN
            user_adj_embedding = model.neighbor_aggregate(user_embedding_input, user_adj_em, torch.LongTensor(user_length_mask).cuda(),user_max_length, torch.FloatTensor(user_adj_td).cuda(), 'user')
            user_ext_embeddings = model.excitement_aggregate(user_adj_em, torch.LongTensor(user_length_mask).cuda(), user_max_length)

        if len(item_adj[itemid]) == 0:
            item_adj_embedding = Variable(torch.zeros(1, model.embedding_dim).cuda()).cuda()
            item_ext_embeddings = Variable(torch.zeros(1, model.final_embedding_dim).cuda()).cuda()
        else:
            item_adj_, item_length_mask, item_max_length = model.adj_sample([item_adj[itemid]], model.sample_length)
            item_adj_em = user_embeddings[torch.LongTensor(item_adj_).cuda(), :].cuda()

            item_time = [[-1 * (timestamp_for_adj - each) / (timestamp_for_adj - item_timestamp_for_adj[itemid][0]) if timestamp_for_adj - item_timestamp_for_adj[itemid][0] != 0 else 1 for each in item_timestamp_for_adj[itemid]]]
            item_adj_td, _, _, = model.adj_sample(item_time, args.sample_length, 'timediffer')

            # time_decay neighbor GCN
            item_adj_embedding = model.neighbor_aggregate(item_embedding_input, item_adj_em, torch.LongTensor(item_length_mask).cuda(), item_max_length, torch.FloatTensor(item_adj_td).cuda(), 'item')
            item_ext_embeddings = model.excitement_aggregate(item_adj_em, torch.LongTensor(item_length_mask).cuda(), item_max_length)

        # symbiotic local learning
        user_local_embedding, item_local_embedding = model.local_aggregate(user_embedding_input, item_embedding_input, user_ext_embeddings, item_ext_embeddings)

        # inherent interaction
        user_inter_embeddings = model.interaction_aggregate(user_embedding_input, item_embedding_input, feature_tensor, user_timediffs_tensor, 'user')
        item_inter_embeddings = model.interaction_aggregate(item_embedding_input, user_embedding_input, feature_tensor, item_timediffs_tensor, 'item')

        # forward with re-scaling network
        user_embedding_output = model.forward(user_embedding_input, user_inter_embeddings, user_local_embedding,
                                              timediffs=user_timediffs_tensor, features=feature_tensor,
                                              adj_embeddings=user_adj_embedding, select='user_update')
        item_embedding_output = model.forward(item_embedding_input, item_inter_embeddings, item_local_embedding,
                                              timediffs=item_timediffs_tensor, features=feature_tensor,
                                              adj_embeddings=item_adj_embedding, select='item_update')

        # Save embeddings
        item_embeddings[itemid,:] = item_embedding_output.squeeze(0)
        user_embeddings[userid,:] = user_embedding_output.squeeze(0)
        user_embeddings_timeseries[j, :] = user_embedding_output.squeeze(0)
        item_embeddings_timeseries[j, :] = item_embedding_output.squeeze(0)

        # UPDATE THE MODEL IN REAL-TIME USING ERRORS MADE IN THE PAST PREDICTION
        if args.online_test:
            loss += args.l2i*MSELoss(item_embedding_output.cuda(), item_embedding_input.cuda().detach())
            loss += args.l2u*MSELoss(user_embedding_output.cuda(), user_embedding_input.cuda().detach())
            
            neg_items = model.sample_for_BPR(userid, num_items-1, [user_adj[userid]], bpr_avoid=None)  # num_items = len(item2Id) + 1
            neg_item_embeddings = item_embeddings[torch.LongTensor(neg_items).cuda(), :]

            # BPR-loss for current t-batch
            # The bpr loss is extremely bigger like 0.79 but each MSE loss is so small like 0.0008
            # So the bpr_coefficient being 0.001 could balance the two task loss
            bpr_loss = model.bpr_loss(user_embedding_output, item_embedding_output, neg_item_embeddings.detach())
            loss += args.bpr_coefficient*bpr_loss

            if timestamp - tbatch_start_time > tbatch_timespan:
                tbatch_start_time = timestamp
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # RESET LOSS FOR NEXT T-BATCH
                loss = 0
        '''
        Note that: 
        we don't re-train the model during test/online stage to keep consistency with the real-world online scenario
        and avoid touching touching ground truth from test data and future information.
        Same operation for baselines methods
        '''

        # current adj should not be allowed to touch future
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

        item_embeddings.detach_()
        user_embeddings.detach_()
        item_embeddings_timeseries.detach_()
        user_embeddings_timeseries.detach_()

    # Performance
    performance_dict = dict()
    ranks = validation_ranks
    mrr = np.mean([1.0 / r for r in ranks])
    rec10 = sum(np.array(ranks) <= 10)*1.0 / len(ranks)
    performance_dict['validation'] = [mrr, rec10]

    ranks = test_ranks
    mrr = np.mean([1.0 / r for r in ranks])
    rec10 = sum(np.array(ranks) <= 10)*1.0 / len(ranks)
    performance_dict['test'] = [mrr, rec10]

    # Print and sava performance
    fw = open(output_fname, "a")
    metrics = ['Mean Reciprocal Rank', 'Recall@10']

    inference_time_per_epoch = time.time()-epoch_start_time
    mean_inference_time_per_epoch.append(inference_time_per_epoch)
    print("\n\n*** {} epoch took {} seconds ***".format(epp, inference_time_per_epoch))
    fw.write('\n\n*** {} epoch took {} seconds ***\n'.format(epp, inference_time_per_epoch))

    print('\n***** Validation performance of epoch %d *****\n' % epp)
    fw.write('\n***** Validation performance of epoch %d *****\n' % epp)
    for i in range(len(metrics)):
        print(metrics[i] + ': ' + str(performance_dict['validation'][i]))
        fw.write("Validation: " + metrics[i] + ': ' + str(performance_dict['validation'][i]) + "\n")

    print('\n***** Test performance of epoch %d *****\n' % epp)
    fw.write('\n***** Test performance of epoch %d *****\n' % epp)
    for i in range(len(metrics)):
        print(metrics[i] + ': ' + str(performance_dict['test'][i]))
        fw.write("Test: " + metrics[i] + ': ' + str(performance_dict['test'][i]) + "\n")
    print('\n')
    fw.flush()
    fw.close()


print("\n\n*** Average time per epoch {} seconds. ***\n\n".format(np.mean(mean_inference_time_per_epoch)))