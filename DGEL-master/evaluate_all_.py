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

def eval_one_epoch(args, model, optimizer, MSELoss, user_sequence_id, item_sequence_id, feature_sequence, item_feat_flag, user_feat_flag, item_feature, user_feature, 
                   user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, timedifference_sequence_for_adj,
                   user_embeddings, item_embeddings, user_embeddings_static, item_embeddings_static, user_embeddings_timeseries, item_embeddings_timeseries,
                   user_adj, item_adj, user_timestamp_for_adj, item_timestamp_for_adj, start_idx, end_idx):
    
    # Performance record
    ranks = []
    inference_time_per_epoch = 0  # 초기화
    
    # 추론 시작 전 메모리 리셋
    torch.cuda.reset_peak_memory_stats()
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
    epoch_start_time = time.time()
    num_items = np.shape(item_embeddings)[0]
    timespan = timestamp_sequence[-1] - timestamp_sequence[0]
    tbatch_timespan = timespan / args.span_num

    for j in range(start_idx, end_idx):

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

        torch.cuda.synchronize()
        epoch_start_time = time.time()

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

        ranks.append(true_item_rank)

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
                item_embeddings.detach_()
                user_embeddings.detach_()
                item_embeddings_timeseries.detach_() 
                user_embeddings_timeseries.detach_()

        else:
            item_embeddings.detach_()
            user_embeddings.detach_()
            item_embeddings_timeseries.detach_() 
            user_embeddings_timeseries.detach_()
        
        
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

        torch.cuda.synchronize()
        inference_time_per_epoch += (time.time()-epoch_start_time)
    # Performance
    # inference_time_per_epoch = time.time()-epoch_start_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    mrr = np.mean([1.0 / r for r in ranks])
    
    #recall_at_k
    rec1 = sum(np.array(ranks) <= 1)*1.0 / len(ranks)
    rec5 = sum(np.array(ranks) <= 5)*1.0 / len(ranks)
    rec10 = sum(np.array(ranks) <= 10)*1.0 / len(ranks)
    rec20 = sum(np.array(ranks) <= 20)*1.0 / len(ranks)
    
    # precision_at_k
    pre1 = sum(np.array(ranks) <= 1) / (len(ranks) * 1)
    pre5 = sum(np.array(ranks) <= 5) / (len(ranks) * 5)
    pre10 = sum(np.array(ranks) <= 10) / (len(ranks) * 10)
    pre20 = sum(np.array(ranks) <= 20) / (len(ranks) * 20)

    ndcg1 = ndcg_at_k(ranks, 1)
    ndcg5 = ndcg_at_k(ranks, 5)
    ndcg10 = ndcg_at_k(ranks, 10)
    ndcg20 = ndcg_at_k(ranks, 20)

    return mrr, rec1, rec5, rec10, rec20, pre1, pre5, pre10, pre20, ndcg1, ndcg5, ndcg10, ndcg20, inference_time_per_epoch, peak_memory_mb

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