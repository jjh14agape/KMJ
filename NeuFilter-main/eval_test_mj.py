import time
import sys
from dataloader import *
from model import *


def eval_one_epoch(args, model, optimizer, MSELoss, user_sequence_id, item_sequence_id, feature_sequence, item_feat_flag, user_feat_flag, item_feature, user_feature, 
                   user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, 
                   user_embeddings, item_embeddings, user_embeddings_static, item_embeddings_static, user_embeddings_timeseries, item_embeddings_timeseries,
                   start_idx, end_idx):

    # validation_ranks = []
    ranks = []
    inference_time_per_epoch = 0  # 초기화

    # 추론 시작 전 메모리 리셋
    torch.cuda.reset_peak_memory_stats()

    tbatch_start_time = None
    loss = 0

    timespan = timestamp_sequence[-1] - timestamp_sequence[0]
    tbatch_timespan = timespan / args.span_num
    device = 'cuda:{}'.format(args.gpu)
    num_items = np.shape(item_embeddings)[0]

    # end_idx = test_start_idx if mode == 'valid' else test_end_idx
    for j in range(start_idx, end_idx):
        # Load  j-th interaction
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

        if not tbatch_start_time:
            tbatch_start_time = timestamp
        itemid_previous = user_previous_itemid_sequence[j]

        torch.cuda.synchronize()
        epoch_start_time = time.time()

        # Load user and item embedding
        user_embedding_input = user_embeddings[torch.LongTensor([userid]).to(device)]
        user_embedding_static_input = user_embeddings_static[torch.LongTensor([userid]).to(device)]
        item_embedding_input = item_embeddings[torch.LongTensor([itemid]).to(device)]
        item_embedding_static_input = item_embeddings_static[torch.LongTensor([itemid]).to(device)]
        feature_tensor = Variable(torch.Tensor(feature).to(device)).unsqueeze(0)
        user_timediffs_tensor = Variable(torch.Tensor([user_timediff]).to(device)).unsqueeze(0)
        item_timediffs_tensor = Variable(torch.Tensor([item_timediff]).to(device)).unsqueeze(0)
        item_embedding_previous = item_embeddings[torch.LongTensor([itemid_previous]).to(device)]

        # Predict user and item embedding
        user_projected_embedding, item_projected_embedding = model.forward(user_embedding_input, item_embedding_input, user_timediffs=user_timediffs_tensor, item_timediffs=item_timediffs_tensor, select='project')
        user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[torch.LongTensor([itemid_previous]).to(device)], user_embedding_static_input], dim=1)

        # Predict embedding of item that user will interact with
        predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

        # Calculate prediction loss
        if args.online_test:
            loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static_input], dim=1).detach())

        # Calculate all items' rank
        euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1), torch.cat([item_embeddings, item_embeddings_static], dim=1)).squeeze(-1)
        
        true_item_distance = euclidean_distances[itemid]
        euclidean_distances_smaller = (euclidean_distances < true_item_distance).data.cpu().numpy()
        true_item_rank = np.sum(euclidean_distances_smaller) + 1
        
        ranks.append(true_item_rank)

        # Update user and item embedding
        user_embedding_output, user_emb_reg = model.forward(user_embedding_input, item_embedding_input, user_prior=user_projected_embedding, users=[userid], features=feature_tensor, select='user_update')
        item_embedding_output, item_emb_reg = model.forward(user_embedding_input, item_embedding_input, item_prior=item_projected_embedding, items=[itemid], features=feature_tensor, select='item_update')

        # user_embedding_output, item_embedding_output = user_projected_embedding, item_projected_embedding

        # Save embeddings
        item_embeddings[itemid,:] = item_embedding_output.squeeze(0).detach()
        user_embeddings[userid,:] = user_embedding_output.squeeze(0).detach()
        user_embeddings_timeseries[j, :] = user_embedding_output.squeeze(0).detach()
        item_embeddings_timeseries[j, :] = item_embedding_output.squeeze(0).detach()

        # Calculate regularization terms
        if args.online_test:
            loss += args.reg_factor1 * MSELoss(item_embedding_output, item_embedding_input.detach())
            loss += args.reg_factor1 * MSELoss(user_embedding_output, user_embedding_input.detach())
            loss += args.reg_factor2 * (user_emb_reg + item_emb_reg)

        # Calculate state change loss
            '''if args.add_state_change_loss:
                loss += calculate_state_prediction_loss(model, [j], user_embeddings_timeseries, y_true, crossEntropyLoss, device)'''

            # update model parameters
            if timestamp - tbatch_start_time > tbatch_timespan:
                tbatch_start_time = timestamp
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss = 0
                item_embeddings.detach_()
                user_embeddings.detach_()
                item_embeddings_timeseries.detach_() 
                user_embeddings_timeseries.detach_()
                model.user_kf.h.detach_()
                model.item_kf.h.detach_()

        else:
            item_embeddings.detach_()
            user_embeddings.detach_()
            item_embeddings_timeseries.detach_() 
            user_embeddings_timeseries.detach_()
            model.user_kf.h.detach_()
            model.item_kf.h.detach_()
        
        torch.cuda.synchronize()
        inference_time_per_epoch += (time.time()-epoch_start_time)

        # t_end = time.time()
            # print('[{}] Processing {}-th/{}-th interactions with loss {:.2f} elapses {:.2f} min'.format('Valid' if mode == 'valid' else 'Test', j, end_idx, loss, (t_end - t_start) / 60.0))

            # Reset loss for next t-batch
        # loss = 0
    
    '''if args.online_test:
        output_fname = "results/interaction_prediction_{}_online_test.txt".format(args.postfix)
    else:
        output_fname = "results/interaction_prediction_{}.txt".format(args.postfix)'''

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
