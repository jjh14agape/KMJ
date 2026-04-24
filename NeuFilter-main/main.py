from train import train
from eval_test import eval_epoch
import numpy as np
import random
import os
import torch
import sys
import argparse

# def set_random_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ['PYTHONHASHSEED'] = str(seed)

def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NeuFilter')
    parser.add_argument('--dataset', default='wikipedia', type=str)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--num_layer_kf', default=2, type=int) # 1,2,3,4,5
    parser.add_argument('--reg_factor1', default=0.01, type=float) #0.001 0.01 0.1 1.0
    parser.add_argument('--reg_factor2', default=0.01, type=float)
    parser.add_argument('--lr', default=0.001, type=float) # [0.0003, 0.001, 0.003]
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--train_proportion', default=0.8, type=float)
    parser.add_argument('--add_state_change_loss', default=False, action='store_true')
    parser.add_argument('--span_num', default=500, type=int, help='time span number')
    # parser.add_argument('--train', default=False, action='store_true')
    # parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--online_test', action='store_true', help='Enable online test mode')
    parser.add_argument('--early_stop', type=int, default=5, help='link or link_sign or sign')
    parser.add_argument('--tolerance', type=float, default=0, help='tolerated marginal improvement for early stopper')

    args = parser.parse_args()
    args.postfix = '{}-{}-{}-{}-{}-{}'.format(args.dataset, args.embedding_dim, args.num_layer_kf, args.reg_factor1, args.reg_factor2, args.lr)

    set_random_seed(args.seed)

    
    print('Start training {} ...'.format(args.dataset))
    train(args)
    print('Training {} successfully'.format(args.dataset))

    '''if args.test == True:
        mean_inference_time_per_epoch = []
        print('Start evaluating {} ...'.format(args.dataset))
        valid_results = []
        for ep in range(args.epochs):
            args.epoch = ep
            eval_epoch(args, mean_inference_time_per_epoch)
            # valid_results.append(valid_result)
        
        print("\n\n*** Average time per epoch {} seconds. ***\n\n".format(np.mean(mean_inference_time_per_epoch)))'''

        # valid_results = np.array(valid_results)
        # best_val_idx = np.argmax(valid_results[:, 0])
        # args.epoch = int(valid_results[best_val_idx, 2])
        # eval_epoch(args, mode='test')
        # print('Evaluating {} successfully'.format(args.dataset))