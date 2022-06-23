import argparse
import json
import logging
import os
import sys
import time
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter

from trainer import GAL_Trainer
from utils import arguments
from utils import utils_models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

sys.path.append(os.getcwd())
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.gal_train_args(parser)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    save_root = os.path.join('checkpoints', 'gal', 'seed_{:d}'.format(args.seed), '{:d}_{:s}{:d}_{:s}'.format(args.model_num, args.arch, args.depth, args.dataset))

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    writer = SummaryWriter(save_root.replace('checkpoints', 'runs'))
    with open(os.path.join(save_root, 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)
    with open(os.path.join(save_root.replace('checkpoints', 'runs'), 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    # set up random seed
    torch.manual_seed(args.seed)

    # initialize models
    models = utils_models.get_models(args, train=True, as_ensemble=False, model_file=None, leaky_relu=True)

    # get data loaders
    trainloader, testloader = utils_models.get_loaders(args, add_gaussian=True)

    # train the ensemble
    trainer = GAL_Trainer(models, trainloader, testloader, writer, save_root, **vars(args))
    trainer.run()

    elapsed = (time.time() - start)
    print('TotalTime: ', elapsed)
