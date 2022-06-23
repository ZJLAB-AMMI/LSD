import os
import sys

sys.path.append(os.getcwd())
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)
import torch.nn as nn

from utils.utils_eval import get_adversary_obj
import argparse
import logging
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

sys.path.append(os.getcwd())
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

import random
import torch
import pickle
from utils import arguments
from utils import utils_models
from utils.utils_eval import get_root_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial examples under the PGD attack method with different perturbation budgets eps', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.dverge_train_args(parser)
    arguments.lcm_gal_train_args(parser)
    arguments.wbox_eval_args(parser)
    args = parser.parse_args()
    args.__setattr__('start_from', 'scratch')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model_num_list = [3]
    attack_list = ['pgd']
    loss_fn = nn.CrossEntropyLoss()
    for dataset in ['cifar10', 'fmnist', 'mnist']:
        random.seed(0)
        torch.manual_seed(0)
        input_file = os.path.join(os.path.join(args.data_dir, 'pick_out_correctly_idx'), 'idx_evaluation_{}.pkl'.format(args.dataset))
        with open(input_file, "rb") as tf:
            subset_correctly_idx = pickle.load(tf)

        subset_idx = subset_correctly_idx[1:2]

        args.__setattr__('dataset', dataset)
        if dataset == 'cifar10':
            args.__setattr__('arch', 'ResNet')
            args.__setattr__('depth', 20)
            args.__setattr__('epochs', 200)

            alg_name_arr = ['lcm_gal_4.0_2.0_4.0']
            eps_list = [0.01, 0.02, 0.03, 0.04]
        elif dataset == 'fmnist':
            args.__setattr__('arch', 'LeNet')
            args.__setattr__('depth', 5)
            args.__setattr__('epochs', 20)
            alg_name_arr = ['lcm_gal_3.0_4.0_4.0']
            eps_list = [0.08, 0.1, 0.15, 0.2]
        else:
            args.__setattr__('arch', 'LeNet')
            args.__setattr__('depth', 5)
            args.__setattr__('epochs', 20)
            alg_name_arr = ['lcm_gal_3.0_4.0_4.0']
            eps_list = [0.1, 0.15, 0.2, 0.25]

        testloader = utils_models.get_testloader(args, batch_size=1, shuffle=False, subset_idx=subset_idx)

        for attack_type in attack_list:
            for alg_name in alg_name_arr:
                for model_num in model_num_list:
                    args.__setattr__('model_num', model_num)
                    args.__setattr__('model_file', os.path.join(get_root_path(args, alg_name), 'epoch_%d.pth' % args.epochs))
                    ensemble = utils_models.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=False)

                    for data, label in testloader:
                        data, label = data.to("cuda"), label.to("cuda")
                        plt.figure(num=0, figsize=(16, 6), dpi=120)  # 设置图形

                        for i, eps in enumerate(eps_list):
                            if i == 0:
                                plt.subplot(1, 5, i + 1)
                                data_0 = torch.squeeze(data.detach(), 0).cpu()

                                if args.arch.lower() == 'resnet':
                                    pic = data_0.reshape(3, 32, 32)
                                else:
                                    pic = data_0.reshape(1, 28, 28)

                                pic = np.transpose(pic.detach().numpy(), (1, 2, 0))
                                plt.imshow(pic)
                                font = {'weight': 'normal', 'size': 30}
                                plt.xlabel(r'$\epsilon$={}'.format(0), font)

                            plt.subplot(1, 5, i + 2)
                            adversary = get_adversary_obj(args, ensemble, loss_fn, eps, attack_type)
                            adv = adversary.perturb(data, label)
                            adv = torch.squeeze(adv, 0).cpu()

                            if args.arch.lower() == 'resnet':
                                pic = adv.reshape(3, 32, 32)
                            else:
                                pic = adv.reshape(1, 28, 28)

                            pic = np.transpose(pic.detach().numpy(), (1, 2, 0))
                            plt.imshow(pic)
                            font = {'weight': 'normal', 'size': 30}
                            plt.xlabel(r'$\epsilon$={}'.format(eps), font)

                        plt.subplot(1, 5, 3)
                        plt.title('Adversarial examples under the {} attack method with different perturbation budgets $\epsilon$.'.format(attack_type.upper()))
                        plt.savefig('{}_{}_{}.jpg'.format(args.dataset, alg_name, attack_type))
                        plt.show()
