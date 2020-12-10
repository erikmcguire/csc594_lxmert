# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument("--train",
                        default='train')
    parser.add_argument("--valid",
                        default='minival')
    parser.add_argument("--test",
                        default=None)
    parser.add_argument('--do_train',
                        type=bool,
                        default=None)
    parser.add_argument('--do_eval',
                        type=bool,
                        default=None)
    parser.add_argument('--do_predict',
                        type=bool,
                        default=None)
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed')
    parser.add_argument('--overwrite_output_dir',
                        default=False)

    # Parameters
    parser.add_argument('--chkpt',
                        type=str,
                        default='2000')
    parser.add_argument('--layer_handling',
                         type=str,
                         default='last',
                         help='last, avg, every')
    parser.add_argument('--head_handling',
                         type=str,
                         default='max',
                         help='last, avg, max')
    parser.add_argument('--ablation',
                         type=str,
                         default='hat',
                         help='Supervision with `random` or human (`hat`).')
    parser.add_argument('--batchSize',
                        dest='batch_size',
                        type=int,
                        default=32)
    parser.add_argument('--topk',
                        type=int,
                        default=None)
    parser.add_argument('--chunkSize',
                         type=int,
                         default=100)
    parser.add_argument('--save_steps',
                         type=int,
                         default=5000)
    parser.add_argument('--lr',
                         type=float,
                         default=1e-4,
                         help='Learning rate.')
    parser.add_argument('--epochs',
                         type=int,
                         default=4,
                         help='Number of epochs.')
    parser.add_argument('--eval_strat',
                        type=str,
                        default='epoch',
                        help='Trainer evaluation strategy: epochs/steps.')
    parser.add_argument('--dropout',
                         type=float,
                         default=0.1)
    parser.add_argument('--x_lmbda',
                         type=float,
                         default=0.5)

    # Loading, Saving
    parser.add_argument('--load',
                         type=str,
                         default=None,
                         help='Load (e.g. fine-tuned) model from path.')
    parser.add_argument('--study_name',
                        type=str,
                        default='vqa')
    parser.add_argument('--save',
                        type=bool,
                        default=None),
    parser.add_argument('--load_best',
                        type=bool,
                        default=None),
    parser.add_argument('--train_type',
                        type=str,
                        default='grid')
    parser.add_argument('--n_trials',
                         type=int,
                         default=20)
    parser.add_argument('--eval_steps',
                         type=int,
                         default=1000)
    parser.add_argument('--output_dir',
                         default='/../content/results')
    parser.add_argument('--ext',
                         default='tsv')
    parser.add_argument('--write',
                         default='nowrite')
    parser.add_argument('--save_att',
                        type=bool,
                        default=None)
    parser.add_argument('--save_preds',
                        type=bool,
                        default=None)
                        
    # Parse the arguments.
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args

args = parse_args()
