import time
import sys
from dataloader import *
import model as lib
from model import *
import torch.optim as optim
from tqdm import tqdm, trange, tqdm_notebook, tnrange
import datetime
from eval_test_mj import eval_one_epoch

def train(args):
    now = datetime.datetime.now()
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
    # args.user_feature_path = "./DyRec/dataset/%s/user_feat.csv" % (args.dataset)
    # args.item_feature_path = "./DyRec/dataset/%s/item_feat.csv" % (args.dataset)
    # log_path = 'DyRec/Our-v2/log/'
    # best_model_root = 'DyRec/Our-v2/best_models/'
    # checkpoint_root = 'DyRec/Our-v2/saved_checkpoints/'

    device = 'cuda:{}'.format(args.gpu) if args.gpu >= 0 else 'cpu'

    logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys.argv, now, log_path, checkpoint_root, best_model_root)

    # # Load dataset
    # [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
    #  item2id, item_sequence_id, item_timediffs_sequence, timestamp_sequence, feature_sequence, y_true] = load_network(args)

    # LOAD DATA
    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
    item2id, item_sequence_id, item_timediffs_sequence,timestamp_sequence, 
    feature_sequence, rating_sequence, sign_sequence, y_true] = load_network(args)


    num_interactions = len(user_sequence_id)
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
    
    # num_features = len(feature_sequence[0]) if args.dataset != 'video' else 0
    true_labels_ratio = len(y_true)/(1.0+sum(y_true))
    train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
    test_start_idx = int(num_interactions * (args.train_proportion+0.1)) # mj
    test_end_idx = int(num_interactions * (args.train_proportion+0.2)) # mj
    timespan = timestamp_sequence[-1] - timestamp_sequence[0]
    tbatch_timespan = timespan / args.span_num

    # Initialize model
    model = NeuFilter(args, num_features, num_users, num_items, device).to(device)
    weight = torch.Tensor([1,true_labels_ratio]).to(device)
    crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
    MSELoss = nn.MSELoss()

    initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).to(device), dim=0))
    initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).to(device), dim=0))
    model.initial_user_embedding = initial_user_embedding
    model.initial_item_embedding = initial_item_embedding

    user_embeddings = initial_user_embedding.repeat(num_users, 1)  # initialize all users to the same embedding
    item_embeddings = initial_item_embedding.repeat(num_items, 1)  # initialize all items to the same embedding
    item_embedding_static = Variable(torch.eye(num_items).to(device))  # one-hot vectors for static embeddings
    user_embedding_static = Variable(torch.eye(num_users).to(device))  # one-hot vectors for static embeddings

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    early_stopper = EarlyStopMonitor(max_round=args.early_stop, tolerance=args.tolerance) 

    is_first_epoch = True
    cached_tbatches_user = {}
    cached_tbatches_item = {}
    cached_tbatches_interactionids = {}
    cached_tbatches_feature = {}
    cached_tbatches_user_timediffs = {}
    cached_tbatches_item_timediffs = {}
    cached_tbatches_previous_item = {}

    mean_learning_time_per_epoch = []
    mean_inference_time_per_epoch = []

    for ep in range(args.epochs):
        user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).to(device))
        item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).to(device))

        optimizer.zero_grad()
        reinitialize_tbatches()
        total_loss, loss, total_interaction_count = 0, 0, 0

        tbatch_start_time = None

        # Train till the end of training interaction idx
        torch.cuda.reset_peak_memory_stats() # 학습 시작 전 메모리 리셋
        torch.cuda.synchronize()
        epoch_start_time = time.time()
        for j in range(train_end_idx):
            if is_first_epoch:
                # Load j-th interaction
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

                # Create t-batch
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

            timestamp = timestamp_sequence[j]
            if tbatch_start_time is None:
                tbatch_start_time = timestamp

            # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
            if timestamp - tbatch_start_time > tbatch_timespan:
                # Reset start time for the next t-batch
                tbatch_start_time = timestamp

                # ITERATE OVER ALL T-BATCHES
                if not is_first_epoch:
                    lib.current_tbatches_user = cached_tbatches_user[timestamp]
                    lib.current_tbatches_item = cached_tbatches_item[timestamp]
                    lib.current_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                    lib.current_tbatches_feature = cached_tbatches_feature[timestamp]
                    lib.current_tbatches_user_timediffs = cached_tbatches_user_timediffs[timestamp]
                    lib.current_tbatches_item_timediffs = cached_tbatches_item_timediffs[timestamp]
                    lib.current_tbatches_previous_item = cached_tbatches_previous_item[timestamp]

                for i in range(len(lib.current_tbatches_user)):
                    # progress_bar3.set_description('Processed %d of %d T-batches ' % (i, len(lib.current_tbatches_user)))
                    total_interaction_count += len(lib.current_tbatches_interactionids[i])

                    # LOAD THE CURRENT TBATCH
                    if is_first_epoch:
                        lib.current_tbatches_user[i] = torch.LongTensor(lib.current_tbatches_user[i]).to(device)
                        lib.current_tbatches_item[i] = torch.LongTensor(lib.current_tbatches_item[i]).to(device)
                        lib.current_tbatches_interactionids[i] = torch.LongTensor(lib.current_tbatches_interactionids[i]).to(device)
                        lib.current_tbatches_feature[i] = torch.Tensor(lib.current_tbatches_feature[i]).to(device)

                        lib.current_tbatches_user_timediffs[i] = torch.Tensor(lib.current_tbatches_user_timediffs[i]).to(device)
                        lib.current_tbatches_item_timediffs[i] = torch.Tensor(lib.current_tbatches_item_timediffs[i]).to(device)
                        lib.current_tbatches_previous_item[i] = torch.LongTensor(lib.current_tbatches_previous_item[i]).to(device)

                    tbatch_userids = lib.current_tbatches_user[i] # Recall "lib.current_tbatches_user[i]" has unique elements
                    tbatch_itemids = lib.current_tbatches_item[i] # Recall "lib.current_tbatches_item[i]" has unique elements
                    tbatch_interactionids = lib.current_tbatches_interactionids[i]
                    feature_tensor = Variable(lib.current_tbatches_feature[i]) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                    user_timediffs_tensor = Variable(lib.current_tbatches_user_timediffs[i]).unsqueeze(1)
                    item_timediffs_tensor = Variable(lib.current_tbatches_item_timediffs[i]).unsqueeze(1)
                    tbatch_itemids_previous = lib.current_tbatches_previous_item[i]
                    item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]

                    # PROJECT USER EMBEDDING TO CURRENT TIME
                    user_embedding_input, item_embedding_input = user_embeddings[tbatch_userids,:], item_embeddings[tbatch_itemids,:]
                    user_projected_embedding, item_projected_embedding = model.forward(user_embedding_input, item_embedding_input, user_timediffs=user_timediffs_tensor, item_timediffs=item_timediffs_tensor, select='project')
                    user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous,:], user_embedding_static[tbatch_userids,:]], dim=1)

                    # PREDICT NEXT ITEM EMBEDDING
                    predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                    # CALCULATE PREDICTION LOSS
                    loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]], dim=1).detach())

                    # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                    user_embedding_output, user_emb_reg = model.forward(user_embedding_input, item_embedding_input, user_prior=user_projected_embedding, users=tbatch_userids, features=feature_tensor, select='user_update')
                    item_embedding_output, item_emb_reg = model.forward(user_embedding_input, item_embedding_input, item_prior=item_projected_embedding, items=tbatch_itemids, features=feature_tensor, select='item_update')

                    # user_embedding_output, item_embedding_output = user_projected_embedding, item_projected_embedding

                    item_embeddings[tbatch_itemids,:] = item_embedding_output
                    user_embeddings[tbatch_userids,:] = user_embedding_output

                    user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                    item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output

                    # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                    loss += args.reg_factor1 * MSELoss(item_embedding_output, item_embedding_input.detach())
                    loss += args.reg_factor1 * MSELoss(user_embedding_output, user_embedding_input.detach())
                    loss += args.reg_factor2 * (user_emb_reg + item_emb_reg)

                    # CALCULATE STATE CHANGE LOSS
                    if args.add_state_change_loss:
                        loss += calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_timeseries, y_true, crossEntropyLoss, device)

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
                model.user_kf.h.detach_()
                model.item_kf.h.detach_()

                # REINITIALIZE
                if is_first_epoch:
                    cached_tbatches_user[timestamp] = lib.current_tbatches_user
                    cached_tbatches_item[timestamp] = lib.current_tbatches_item
                    cached_tbatches_interactionids[timestamp] = lib.current_tbatches_interactionids
                    cached_tbatches_feature[timestamp] = lib.current_tbatches_feature
                    cached_tbatches_user_timediffs[timestamp] = lib.current_tbatches_user_timediffs
                    cached_tbatches_item_timediffs[timestamp] = lib.current_tbatches_item_timediffs
                    cached_tbatches_previous_item[timestamp] = lib.current_tbatches_previous_item

                    reinitialize_tbatches()
            t_end = time.time()
            # print('[Train] Processing {}th/{}th interactions in {}th/{}th epoch elapses {:.2f} min'.format(j, train_end_idx, ep, args.epochs, (t_end-t_start)/60.0))

        is_first_epoch = False
        torch.cuda.synchronize()
        learning_time_per_epoch = time.time()-epoch_start_time

        # 학습 피크 메모리 측정
        train_peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        train_current_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        if ep > 0:
            mean_learning_time_per_epoch.append(learning_time_per_epoch)
        # print("\n\n{} epoch took {} seconds".format(ep, learning_time_per_epoch))
        # End of epoch
        # print("\nTotal loss in this epoch = %f" % (total_loss))
        val_mrr, val_rec1, val_rec5, val_rec10, val_rec20, val_pre1, val_pre5, val_pre10, val_pre20, val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20, inference_time_per_epoch, val_peak_memory_mb  = eval_one_epoch(args, model, optimizer, MSELoss, user_sequence_id, item_sequence_id, feature_sequence, item_feat_flag, user_feat_flag, item_feature, user_feature, 
                   user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, 
                   user_embeddings, item_embeddings, user_embedding_static, item_embedding_static, user_embeddings_timeseries, item_embeddings_timeseries,
                   validation_start_idx, test_start_idx)

        mean_inference_time_per_epoch.append(inference_time_per_epoch)
        # End of epoch
        # 같은 메트릭끼리 한 줄에 탭으로 구분하여 출력
        logger.info('epoch: {}'.format(ep))
        logger.info('Total loss in this epoch: {}'.format(total_loss))
        logger.info('val MRR: {:.4f}'.format(val_mrr))
        logger.info('val REC:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_rec1, val_rec5, val_rec10, val_rec20))
        logger.info('val PRE:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_pre1, val_pre5, val_pre10, val_pre20))
        logger.info('val NDCG:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(val_ndcg1, val_ndcg5, val_ndcg10, val_ndcg20))
        logger.info('Trainig time: {} sec'.format(learning_time_per_epoch))
        logger.info('Inference time: {} sec'.format(inference_time_per_epoch))
        logger.info('Training peak memory: {:.2f} MiB (current: {:.2f} MiB)'.format(train_peak_memory_mb, train_current_memory_mb))  # 추가
        logger.info('Inference peak memory: {:.2f} MiB'.format(val_peak_memory_mb))  # 추가


        if early_stopper.early_stop_check(val_mrr):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_checkpoint_path = get_checkpoint_path(early_stopper.best_epoch)
            model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, best_checkpoint_path, device)
            set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx)

            item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
            item_embeddings = item_embeddings.clone()
            item_embedding_static = item_embeddings_dystat[:, args.embedding_dim:]
            item_embedding_static = item_embedding_static.clone()

            user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
            user_embeddings = user_embeddings.clone()
            user_embedding_static = user_embeddings_dystat[:, args.embedding_dim:]
            user_embedding_static = user_embedding_static.clone()

            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            if args.online_test == False:
                model.eval()
            break
        else:

            item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
            user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
            
            save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries, get_checkpoint_path(ep))

            user_embeddings = initial_user_embedding.repeat(num_users, 1)
            item_embeddings = initial_item_embedding.repeat(num_items, 1)

    # print("\n***** Training complete. Saving final model. *****\n")
    # print("\n\n*** Average time per epoch {} seconds. ***\n\n".format(np.mean(mean_learning_time_per_epoch)))

    # save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)

    # Save final model of final epoch
    logger.info("\n***** Training complete. Testing final model. *****\n")
    test_mrr, test_rec1, test_rec5, test_rec10, test_rec20, test_pre1, test_pre5, test_pre10, test_pre20, test_ndcg1, test_ndcg5, test_ndcg10, test_ndcg20, inference_time, test_peak_memory_mb = eval_one_epoch(args, model, optimizer, MSELoss, user_sequence_id, item_sequence_id, feature_sequence, item_feat_flag, user_feat_flag, item_feature, user_feature, 
                   user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence, 
                   user_embeddings, item_embeddings, user_embedding_static, item_embedding_static, user_embeddings_timeseries, item_embeddings_timeseries,
                   test_start_idx, test_end_idx)

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