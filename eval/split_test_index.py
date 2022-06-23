import argparse
import logging
import os
import pickle
import random
import sys
import warnings

sys.path.append(os.getcwd())
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

import torch

from utils import arguments
from utils import utils_models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split_test_index', add_help=True)

    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.lcm_gal_train_args(parser)
    arguments.transf_eval_args(parser)
    args = parser.parse_args()
    args.__setattr__('start_from', 'scratch')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    testloader = utils_models.get_testloader(args, batch_size=args.batch_size)
    random.seed(0)
    torch.manual_seed(0)

    output_root = os.path.join(args.data_dir, 'pick_out_correctly_idx')
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    subset_idx = random.sample(range(len(testloader.dataset)), len(testloader.dataset))

    subset_idx_para_compare = subset_idx[0:1000]
    output_file = os.path.join(output_root, 'idx_para_compare_{}.pkl'.format(args.dataset))
    with open(output_file, "wb") as tf:
        pickle.dump(subset_idx_para_compare, tf)

    subset_idx_test = subset_idx[1000:]
    output_file = os.path.join(output_root, 'idx_evaluation_{}.pkl'.format(args.dataset))
    with open(output_file, "wb") as tf:
        pickle.dump(subset_idx_test, tf)
