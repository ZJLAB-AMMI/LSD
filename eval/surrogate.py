from __future__ import print_function

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

from utils.utils_eval import get_adversary_obj

import torch
import torch.nn as nn
import torch.nn.functional as F
from advertorch.attacks.utils import attack_whole_dataset
from advertorch.utils import to_one_hot
from tqdm import tqdm

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


class CarliniWagnerLoss(nn.Module):
    def __init__(self, conf=50.):
        super(CarliniWagnerLoss, self).__init__()
        self.conf = conf

    def forward(self, input, target):
        num_classes = input.size(1)
        label_mask = to_one_hot(target, num_classes=num_classes).float()
        correct_logit = torch.sum(label_mask * input, dim=1)
        wrong_logit = torch.max((1. - label_mask) * input, dim=1)[0]
        loss = -F.relu(correct_logit - wrong_logit + self.conf).sum()
        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='surrogate', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.bbox_eval_args(parser)
    args = parser.parse_args()

    if args.dataset.lower() == 'cifar10':
        eps_list = [0.01, 0.02, 0.03, 0.04]
    elif args.dataset.lower() == 'fmnist':
        eps_list = [0.08, 0.1, 0.15, 0.2]
    else:
        eps_list = [0.1, 0.15, 0.2, 0.25]
    model_list = [3, 5]
    attack_list = ['bim', 'fgsm', 'mim', 'pgd']

    surrogate = []
    for num_models in model_list:
        save_root = os.path.join('./checkpoints', 'baseline', 'seed_{:d}'.format(args.seed), '{:d}_{:s}{:d}_{:s}'.format(num_models, args.arch, args.depth, args.dataset))
        ensemble = utils_models.get_models(args, train=False, as_ensemble=True, model_file=os.path.join(save_root, 'epoch_%d.pth' % args.epochs), leaky_relu=False)
        surrogate.append(ensemble)

    # ---------------------------------------------------------------------------------------------------------------------------------------------------surrogate test
    random.seed(0)
    torch.manual_seed(0)
    input_file = os.path.join(os.path.join(args.data_dir, 'pick_out_correctly_idx'), 'idx_evaluation_{}.pkl'.format(args.dataset))
    with open(input_file, "rb") as tf:
        subset_correctly_idx = pickle.load(tf)

    testloader = utils_models.get_testloader(args, batch_size=200, shuffle=False, subset_idx=subset_correctly_idx)

    for attack_type in attack_list:
        attack_name = attack_type.upper()
        for eps in eps_list:
            transfer = []
            for curmodel in surrogate:
                for i in range(2):
                    for j in range(3):
                        random.seed(j)
                        torch.manual_seed(j)
                        loss_fn = nn.CrossEntropyLoss() if (i == 1) else CarliniWagnerLoss(conf=.1)
                        adversary = get_adversary_obj(args, curmodel, loss_fn, eps, attack_type)
                        test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)
                        _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")
                        _ = _.cpu().detach().numpy()
                        transfer.append(_)
            print('generating adv_examples_{}_{}'.format(attack_name, str(int(255 * eps))))

            output_root = os.path.join(args.data_dir, args.folder)
            if not os.path.exists(output_root):
                os.makedirs(output_root)

            output_file = os.path.join(output_root, '{}_adv_examples_eps_{}_{}.pkl'.format(attack_name, str(int(255 * eps)), args.dataset))
            with open(output_file, "wb") as tf:
                pickle.dump(transfer, tf)

    # ---------------------------------------------------------------------------------------------------------------------------------------------------surrogate para_compare
    random.seed(0)
    torch.manual_seed(0)
    input_file = os.path.join(os.path.join(args.data_dir, 'pick_out_correctly_idx'), 'idx_para_compare_{}.pkl'.format(args.dataset))
    with open(input_file, "rb") as tf:
        subset_correctly_idx = pickle.load(tf)

    testloader = utils_models.get_testloader(args, batch_size=200, shuffle=False, subset_idx=subset_correctly_idx)

    for attack_type in attack_list:
        attack_name = attack_type.upper()
        for eps in eps_list:
            transfer = []
            for curmodel in surrogate:
                for i in range(2):
                    for j in range(3):
                        random.seed(j)
                        torch.manual_seed(j)
                        loss_fn = nn.CrossEntropyLoss() if (i == 1) else CarliniWagnerLoss(conf=.1)
                        adversary = get_adversary_obj(args, curmodel, loss_fn, eps, attack_type)
                        test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)
                        _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")
                        _ = _.cpu().detach().numpy()
                        transfer.append(_)
            print('generating adv_examples_{}_{}'.format(attack_name, str(int(255 * eps))))

            output_root = os.path.join(args.data_dir, args.folder)
            if not os.path.exists(output_root):
                os.makedirs(output_root)

            output_file = os.path.join(output_root, '{}_adv_examples_eps_{}_{}_para_compare.pkl'.format(attack_name, str(int(255 * eps)), args.dataset))
            with open(output_file, "wb") as tf:
                pickle.dump(transfer, tf)
