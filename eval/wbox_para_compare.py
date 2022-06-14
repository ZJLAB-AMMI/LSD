import argparse
import logging
import os
import pickle
import random
import sys
import time
import warnings

sys.path.append(os.getcwd())
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)
import pandas as pd
import torch
import torch.nn as nn
from advertorch.attacks.utils import attack_whole_dataset
from tqdm import tqdm

from utils import arguments
from utils.utils_eval import get_adversary_obj
from utils import utils_models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of White-box Robustness of Ensembles with different hyper-parameter settings', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.dverge_train_args(parser)
    arguments.lcm_gal_train_args(parser)
    arguments.wbox_eval_args(parser)
    args = parser.parse_args()
    args.__setattr__('start_from', 'scratch')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    batch_size = 200

    attack_list = ['bim', 'fgsm', 'mim', 'pgd']  #
    if args.dataset.lower() == 'cifar10':
        ld_coeff_list = [4.0]
        alpha_list = [0.5, 1.0, 2.0, 4.0]
        beta_list = [0.0, 0.5, 1.0, 2.0, 4.0]
        eps_list = [0.03, 0.04]
    elif args.dataset.lower() == 'fmnist':
        ld_coeff_list = [3.0]
        alpha_list = [0.5, 1.0, 2.0, 4.0]
        beta_list = [0.0, 0.5, 1.0, 2.0, 4.0]
        eps_list = [0.08, 0.1, 0.15, 0.2]
    else:
        ld_coeff_list = [3.0]
        alpha_list = [0.5, 1.0, 2.0, 4.0]
        beta_list = [0.0, 0.5, 1.0, 2.0, 4.0]
        eps_list = [0.2, 0.25]

    model_num = 3
    random.seed(0)
    torch.manual_seed(0)

    input_file = os.path.join(os.path.join(args.data_dir, 'pick_out_correctly_idx'), 'idx_para_compare_{}.pkl'.format(args.dataset))
    with open(input_file, "rb") as tf:
        subset_correctly_idx = pickle.load(tf)
    testloader = utils_models.get_testloader(args, batch_size=200, shuffle=False, subset_idx=subset_correctly_idx)

    output_root = os.path.join('./results/para_compare', 'wbox', str(args.seed))
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    output_file_csv = os.path.join(output_root, '{}_wbox_para_compare_results_{}_{}.csv'.format(args.dataset, 'adv' if args.plus_adv else 'clean', time.strftime('%Y_%m_%d_%H_%M')))

    loss_fn = nn.CrossEntropyLoss()

    # dict init
    rob = {}
    for ld_coeff in ld_coeff_list:
        coeff_name = 'ld_coeff_' + str(ld_coeff)
        rob[coeff_name] = {}

        for alpha in alpha_list:
            alpha_name = 'alpha_' + str(alpha)
            rob[coeff_name][alpha_name] = {}

            for beta in beta_list:
                beta_name = 'beta_' + str(beta)
                rob[coeff_name][alpha_name][beta_name] = {}

                for attack_type in attack_list:
                    rob[coeff_name][alpha_name][beta_name][attack_type] = {}
                    for eps in eps_list:
                        eps_name = 'eps_' + str(int(eps * 255))
                        rob[coeff_name][alpha_name][beta_name][attack_type][eps_name] = 0.0

    for ld_coeff in ld_coeff_list:
        coeff_name = 'ld_coeff_' + str(ld_coeff)

        for alpha in alpha_list:
            alpha_name = 'alpha_' + str(alpha)

            for beta in beta_list:
                beta_name = 'beta_' + str(beta)

                save_root = os.path.join('checkpoints/', 'lcm_gal_{:.1f}_{:.1f}_{:.1f}'.format(ld_coeff, alpha, beta), 'seed_{:d}'.format(args.seed),
                                         '{:d}_{:s}{:d}_{:s}'.format(model_num, args.arch, args.depth, args.dataset))
                args.__setattr__('model_file', os.path.join(save_root, 'epoch_%d.pth' % args.epochs))

                leaky_relu = True if 'gal' in args.model_file else False
                ensemble = utils_models.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=leaky_relu)

                for attack_type in attack_list:
                    tqdm.write('attack_type:{}'.format(attack_type))

                    for eps in eps_list:
                        eps_name = 'eps_' + str(int(eps * 255))

                        correct_or_not = []
                        for i in range(args.random_start):
                            random.seed(i)
                            torch.manual_seed(i)
                            adversary = get_adversary_obj(args, ensemble, loss_fn, eps, attack_type)
                            _, label, pred, advpred = attack_whole_dataset(adversary, testloader, device="cuda")
                            correct_or_not.append(label == advpred)

                        correct_or_not = torch.stack(correct_or_not, dim=-1).all(dim=-1)
                        acc = 100. * (label == pred).sum().item() / len(label)
                        attack_acc = 100. * correct_or_not.sum().item() / len(label)
                        rob[coeff_name][alpha_name][beta_name][attack_type][eps_name] = attack_acc
                        tqdm.write('attack: {}\t coeff: {}\t alpha: {}\t beta: {}\t eps: {}\t attack_acc: {}'.format(attack_type, coeff_name, alpha_name, beta_name, eps_name, attack_acc))

    # dict to dict
    rob_csv = {}
    for attack_type in attack_list:
        rob_csv[attack_type] = {}
        for eps in eps_list:
            eps_name = 'eps_' + str(int(eps * 255))
            rob_csv[attack_type][eps_name] = {}
            for ld_coeff in ld_coeff_list:
                coeff_name = 'ld_coeff_' + str(ld_coeff)
                rob_csv[attack_type][eps_name][coeff_name] = {}

                for alpha in alpha_list:
                    alpha_name = 'alpha_' + str(alpha)
                    rob_csv[attack_type][eps_name][coeff_name][alpha_name] = {}

                    for beta in beta_list:
                        beta_name = 'beta_' + str(beta)
                        rob_csv[attack_type][eps_name][coeff_name][alpha_name][beta_name] = rob[coeff_name][alpha_name][beta_name][attack_type][eps_name]

    # dict to pd
    rob_pd_all = None
    for key_0, value_0 in rob_csv.items():
        for key_1, value_1 in value_0.items():
            for key_2, value_2 in value_1.items():
                rob_pd = pd.DataFrame(data=value_2).T
                rob_pd.insert(loc=0, column="alpha", value=alpha_list)
                rob_pd.insert(loc=0, column="ld_coeff", value=key_2.split('_')[-1])
                rob_pd.insert(loc=0, column="eps", value=key_1.split('_')[-1])
                rob_pd.insert(loc=0, column="attack_type", value=key_0)

                rob_pd_all = pd.concat([rob_pd_all, rob_pd])
    print(rob_pd_all)

    # save to file
    rob_pd_all.to_csv(output_file_csv, sep=',', index=False, float_format='%.2f')
