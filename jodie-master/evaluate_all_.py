'''
This code evaluates the validation and test performance in an epoch of the model trained in jodie.py.
The task is: interaction prediction, i.e., predicting which item will a user interact with? 

To calculate the performance for one epoch:
$ python evaluate_interaction_prediction.py --network reddit --model jodie --epoch 49

To calculate the performance for all epochs, use the bash file, evaluate_all_epochs.sh, which calls this file once for every epoch.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

from library_data import *
from library_models import *
import time

def eval_one_epoch(args, model, optimizer, MSELoss, crossEntropyLoss, weight, user_sequence_id, item_sequence_id, feature_sequence, item_feat_flag, user_feat_flag, item_feature, user_feature, 
                   user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, 
                   user_embeddings, item_embeddings, user_embeddings_static, item_embeddings_static, user_embeddings_timeseries, item_embeddings_timeseries,
                   start_idx, end_idx, filter_users=None):
    # PERFORMANCE METRICS
    ranks = []
    inference_time_per_epoch = 0  # 초기화
    # 추론 시작 전 메모리 리셋
    torch.cuda.reset_peak_memory_stats()

            # --- 선택적 사용자 필터 (High/Low 버킷 평가용) ---
    '''if filter_users is not None:
        if not isinstance(filter_users, set):
            filter_users = set(filter_users)
        def _use_user(u):
            return (u in filter_users)
    else:
        def _use_user(u):
            return True'''
        

    ''' 
    Here we use the trained model to make predictions for the validation and testing interactions.
    The model does a forward pass from the start of validation till the end of testing.
    For each interaction, the trained model is used to predict the embedding of the item it will interact with. 
    This is used to calculate the rank of the true item the user actually interacts with.

    After this prediction, the errors in the prediction are used to calculate the loss and update the model parameters. 
    This simulates the real-time feedback about the predictions that the model gets when deployed in-the-wild. 
    Please note that since each interaction in validation and test is only seen once during the forward pass, there is no data leakage. 
    '''
    tbatch_start_time = None
    loss = 0
    # FORWARD PASS
    print("*** Making interaction predictions by forward pass (no t-batching) ***")
    # epoch_start_time = time.time()
    num_items = np.shape(item_embeddings)[0]
    timespan = timestamp_sequence[-1] - timestamp_sequence[0]
    tbatch_timespan = timespan / args.span_num

    # print('start time:', datetime.datetime.now())
    # with trange(train_end_idx, test_end_idx) as progress_bar:
    for j in range(start_idx, end_idx):
        # progress_bar.set_description('%dth interaction for validation and testing' % j)
        userid = int(user_sequence_id[j])
        '''if not _use_user(userid): 
            continue'''   # ★ 필터 외 사용자는 스킵
        # LOAD INTERACTION J
        # userid = user_sequence_id[j]
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
        if not tbatch_start_time:
            tbatch_start_time = timestamp
        itemid_previous = user_previous_itemid_sequence[j]

        torch.cuda.synchronize()
        epoch_start_time = time.time()
        # LOAD USER AND ITEM EMBEDDING
        user_embedding_input = user_embeddings[torch.cuda.LongTensor([userid])]
        user_embedding_static_input = user_embeddings_static[torch.cuda.LongTensor([userid])]
        item_embedding_input = item_embeddings[torch.cuda.LongTensor([itemid])]
        item_embedding_static_input = item_embeddings_static[torch.cuda.LongTensor([itemid])]
        feature_tensor = Variable(torch.Tensor(feature).cuda()).unsqueeze(0)
        user_timediffs_tensor = Variable(torch.Tensor([user_timediff]).cuda()).unsqueeze(0)
        item_timediffs_tensor = Variable(torch.Tensor([item_timediff]).cuda()).unsqueeze(0)
        item_embedding_previous = item_embeddings[torch.cuda.LongTensor([itemid_previous])]

        # PROJECT USER EMBEDDING
        user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
        user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[torch.cuda.LongTensor([itemid_previous])], user_embedding_static_input], dim=1)

        # PREDICT ITEM EMBEDDING
        predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

        # CALCULATE PREDICTION LOSS
        if args.online_test:
            loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static_input], dim=1).detach())
        
        # CALCULATE DISTANCE OF PREDICTED ITEM EMBEDDING TO ALL ITEMS 
        euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1), torch.cat([item_embeddings, item_embeddings_static], dim=1)).squeeze(-1) 
        
        # CALCULATE RANK OF THE TRUE ITEM AMONG ALL ITEMS
        true_item_distance = euclidean_distances[itemid]
        euclidean_distances_smaller = (euclidean_distances < true_item_distance).data.cpu().numpy()
        true_item_rank = np.sum(euclidean_distances_smaller) + 1

        ranks.append(true_item_rank)

        # UPDATE USER AND ITEM EMBEDDING
        user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update') 
        item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update') 

        # SAVE EMBEDDINGS
        item_embeddings[itemid,:] = item_embedding_output.squeeze(0) 
        user_embeddings[userid,:] = user_embedding_output.squeeze(0) 
        user_embeddings_timeseries[j, :] = user_embedding_output.squeeze(0)
        item_embeddings_timeseries[j, :] = item_embedding_output.squeeze(0)
        
        # UPDATE THE MODEL IN REAL-TIME USING ERRORS MADE IN THE PAST PREDICTION
        if args.online_test:
            # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
            loss += MSELoss(item_embedding_output, item_embedding_input.detach())
            loss += MSELoss(user_embedding_output, user_embedding_input.detach())
            # CALCULATE STATE CHANGE LOSS
            # if args.state_change:
            #     loss += calculate_state_prediction_loss(model, [j], user_embeddings_timeseries, y_true, crossEntropyLoss) 


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
        
        torch.cuda.synchronize()
        inference_time_per_epoch += (time.time()-epoch_start_time)

    # CALCULATE THE PERFORMANCE METRICS
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