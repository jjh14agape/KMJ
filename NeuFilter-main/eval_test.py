import time
import sys
from dataloader import *
from model import *


def eval_epoch(args, mean_inference_time_per_epoch):
    args.datapath = "../dataset/%s/%s.csv" % (args.dataset, args.dataset)
    args.user_feature_path = "../dataset/%s/user_feat.csv" % (args.dataset)
    args.item_feature_path = "../dataset/%s/item_feat.csv" % (args.dataset)
    # args.datapath = "./DyRec/dataset/%s/%s.csv" % (args.dataset, args.dataset)
    # user_feature_path = "./DyRec/dataset/%s/user_feat.csv" % (args.dataset)
    # item_feature_path = "./DyRec/dataset/%s/item_feat.csv" % (args.dataset)
    device = 'cuda:{}'.format(args.gpu)

    # load dataset
    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,\
     item2id, item_sequence_id, item_timediffs_sequence,timestamp_sequence, feature_sequence,y_true] = load_network(args)
    
    num_interactions = len(user_sequence_id)
    # num_features = len(feature_sequence[0]) if args.dataset != 'video' else 0
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

    true_labels_ratio = len(y_true)/(sum(y_true)+1)

    train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
    test_start_idx = int(num_interactions * (args.train_proportion + 0.1))
    test_end_idx = int(num_interactions * (args.train_proportion + 0.2))

    timespan = timestamp_sequence[-1] - timestamp_sequence[0]
    tbatch_timespan = timespan / args.span_num

    model = NeuFilter(args, num_features, num_users, num_items, device).to(device)
    weight = torch.Tensor([1,true_labels_ratio]).to(device)
    crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
    MSELoss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, args.epoch, device)

    set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx)
    
    item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
    item_embeddings = item_embeddings.clone()
    item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
    item_embeddings_static = item_embeddings_static.clone()

    user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
    user_embeddings = user_embeddings.clone()
    user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
    user_embeddings_static = user_embeddings_static.clone()

    validation_ranks = []
    test_ranks = []

    tbatch_start_time = None
    loss = 0

    epoch_start_time = time.time()
    # end_idx = test_start_idx if mode == 'valid' else test_end_idx
    for j in range(train_end_idx, test_end_idx):
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
        
        if j < test_start_idx:
            validation_ranks.append(true_item_rank)
        else:
            test_ranks.append(true_item_rank)

        # Update user and item embedding
        user_embedding_output, user_emb_reg = model.forward(user_embedding_input, item_embedding_input, user_prior=user_projected_embedding, users=[userid], features=feature_tensor, select='user_update')
        item_embedding_output, item_emb_reg = model.forward(user_embedding_input, item_embedding_input, item_prior=item_projected_embedding, items=[itemid], features=feature_tensor, select='item_update')

        # user_embedding_output, item_embedding_output = user_projected_embedding, item_projected_embedding

        # Save embeddings
        item_embeddings[itemid,:] = item_embedding_output.squeeze(0)
        user_embeddings[userid,:] = user_embedding_output.squeeze(0)
        user_embeddings_timeseries[j, :] = user_embedding_output.squeeze(0)
        item_embeddings_timeseries[j, :] = item_embedding_output.squeeze(0)

        # Calculate regularization terms
        if args.online_test:
            loss += args.reg_factor1 * MSELoss(item_embedding_output, item_embedding_input.detach())
            loss += args.reg_factor1 * MSELoss(user_embedding_output, user_embedding_input.detach())
            loss += args.reg_factor2 * (user_emb_reg + item_emb_reg)

        # Calculate state change loss
            if args.add_state_change_loss:
                loss += calculate_state_prediction_loss(model, [j], user_embeddings_timeseries, y_true, crossEntropyLoss, device)

            # update model parameters
            if timestamp - tbatch_start_time > tbatch_timespan:
                tbatch_start_time = timestamp
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss = 0

        # t_end = time.time()
            # print('[{}] Processing {}-th/{}-th interactions with loss {:.2f} elapses {:.2f} min'.format('Valid' if mode == 'valid' else 'Test', j, end_idx, loss, (t_end - t_start) / 60.0))

            # Reset loss for next t-batch
        # loss = 0
        item_embeddings.detach_()
        user_embeddings.detach_()
        item_embeddings_timeseries.detach_()
        user_embeddings_timeseries.detach_()
        model.user_kf.h.detach_()
        model.item_kf.h.detach_()
    
    if args.online_test:
        output_fname = "results/interaction_prediction_{}_online_test.txt".format(args.postfix)
    else:
        output_fname = "results/interaction_prediction_{}.txt".format(args.postfix)

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
    print("\n\n*** {} epoch took {} seconds ***".format(args.epoch, inference_time_per_epoch))
    fw.write('\n\n*** {} epoch took {} seconds ***\n'.format(args.epoch, inference_time_per_epoch))

    print('\n***** Validation performance of epoch %d *****\n' % args.epoch)
    fw.write('\n***** Validation performance of epoch %d *****\n' % args.epoch)
    for i in range(len(metrics)):
        print(metrics[i] + ': ' + str(performance_dict['validation'][i]))
        fw.write("Validation: " + metrics[i] + ': ' + str(performance_dict['validation'][i]) + "\n")

    print('\n***** Test performance of epoch %d *****\n' % args.epoch)
    fw.write('\n***** Test performance of epoch %d *****\n' % args.epoch)
    for i in range(len(metrics)):
        print(metrics[i] + ': ' + str(performance_dict['test'][i]))
        fw.write("Test: " + metrics[i] + ': ' + str(performance_dict['test'][i]) + "\n")
    print('\n')
    fw.flush()
    fw.close()

    

