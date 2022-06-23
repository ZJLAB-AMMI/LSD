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
from utils.utils_eval import get_adversary_obj, get_root_path
from utils import utils_models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of White-box Robustness of Ensembles with Advertorch', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.dverge_train_args(parser)
    arguments.lcm_gal_train_args(parser)
    arguments.wbox_eval_args(parser)
    args = parser.parse_args()
    args.__setattr__('start_from', 'scratch')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.dataset.lower() == 'cifar10':
        eps_list = [0.01, 0.02, 0.03, 0.04]
        model_num_list = [3, 5]
        alg_name_arr = ['baseline', 'adp', 'gal', 'dverge', 'trs',
                        'lcm_gal_4.0_0.0_0.0', 'lcm_gal_4.0_1.0_0.0', 'lcm_gal_4.0_2.0_0.0', 'lcm_gal_4.0_4.0_0.0',
                        'lcm_gal_4.0_1.0_2.0', 'lcm_gal_4.0_2.0_2.0', 'lcm_gal_4.0_0.5_4.0', 'lcm_gal_4.0_2.0_4.0']
    elif args.dataset.lower() == 'fmnist':
        eps_list = [0.08, 0.1, 0.15, 0.2]
        model_num_list = [3, 5]
        alg_name_arr = ['baseline', 'adp', 'gal', 'dverge', 'trs',
                        'lcm_gal_3.0_0.0_0.0', 'lcm_gal_3.0_1.0_0.0', 'lcm_gal_3.0_2.0_0.0', 'lcm_gal_3.0_4.0_0.0',
                        'lcm_gal_3.0_1.0_2.0', 'lcm_gal_3.0_1.0_4.0', 'lcm_gal_3.0_2.0_4.0', 'lcm_gal_3.0_4.0_4.0']
    else:
        eps_list = [0.1, 0.15, 0.2, 0.25]
        model_num_list = [3, 5]
        alg_name_arr = ['baseline', 'adp', 'gal', 'dverge', 'trs',
                        'lcm_gal_3.0_0.0_0.0', 'lcm_gal_3.0_1.0_0.0', 'lcm_gal_3.0_2.0_0.0', 'lcm_gal_3.0_4.0_0.0',
                        'lcm_gal_3.0_1.0_2.0', 'lcm_gal_3.0_1.0_4.0', 'lcm_gal_3.0_2.0_4.0', 'lcm_gal_3.0_4.0_4.0', 'lcm_gal_3.0_2.0_1.0']

    attack_list = ['bim', 'fgsm', 'mim', 'pgd']

    random.seed(0)
    torch.manual_seed(0)
    input_file = os.path.join(os.path.join(args.data_dir, 'pick_out_correctly_idx'), 'idx_evaluation_{}.pkl'.format(args.dataset))
    with open(input_file, "rb") as tf:
        subset_correctly_idx = pickle.load(tf)
    testloader = utils_models.get_testloader(args, batch_size=200, shuffle=False, subset_idx=subset_correctly_idx)

    output_root = os.path.join('./results', 'wbox', str(args.seed))
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    output_file_csv = os.path.join(output_root, '{}_wbox_results_{}_{}.csv'.format(args.dataset, 'clean', time.strftime('%Y_%m_%d_%H_%M')))

    loss_fn = nn.CrossEntropyLoss()

    # dict init
    rob = {}
    for _, attack_type in enumerate(attack_list):
        rob[attack_type] = {}
        for _, alg_name in enumerate(alg_name_arr):
            rob[attack_type][alg_name] = {}
            for _, model_num in enumerate(model_num_list):
                en_name = 'emodel_' + str(model_num)
                rob[attack_type][alg_name][en_name] = {}
                for _, eps in enumerate(eps_list):
                    eps_name = 'eps_' + str(int(eps * 255))
                    rob[attack_type][alg_name][en_name][eps_name] = 0.0

    for attack_type in attack_list:
        tqdm.write('attack_type:{}'.format(attack_type))
        for alg_name in alg_name_arr:
            alg_name_t = alg_name

            tqdm.write('alg_name:{}'.format(alg_name))
            for model_num in model_num_list:
                random.seed(0)
                torch.manual_seed(0)
                en_name = 'emodel_' + str(model_num)
                args.__setattr__('model_num', model_num)

                save_root = get_root_path(args, alg_name_t)
                args.__setattr__('model_file', os.path.join(save_root, 'epoch_%d.pth' % args.epochs))

                leaky_relu = True if 'gal' in args.model_file else False
                ensemble = utils_models.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=leaky_relu)

                tqdm.write('model_num:{}'.format(model_num))
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
                    rob[attack_type][alg_name][en_name][eps_name] = attack_acc
                    tqdm.write('attack_type: {}\t alg_name: {}\t en_name: {}\t eps_name: {}\t attack_acc: {}'.format(attack_type, alg_name, en_name, eps_name, attack_acc))

    # dict to dict
    rob_csv = {}
    for k, model_num in enumerate(model_num_list):
        en_name = 'emodel_' + str(model_num)
        rob_csv[en_name] = {}
        for i, attack_type in enumerate(attack_list):
            rob_csv[en_name][attack_type] = {}

            for l, eps in enumerate(eps_list):
                eps_name = 'eps_' + str(int(eps * 255))
                rob_csv[en_name][attack_type][eps_name] = {}

                for j, alg_name in enumerate(alg_name_arr):
                    rob_csv[en_name][attack_type][eps_name][alg_name] = rob[attack_type][alg_name][en_name][eps_name]

    # dict to pd
    rob_pd_all = None
    for key_0, value_0 in rob_csv.items():
        for key_1, value_1 in value_0.items():
            rob_pd = pd.DataFrame(data=value_1).T
            rob_pd.insert(loc=0, column="eps", value=eps_list)
            rob_pd.insert(loc=0, column="attack_type", value=key_1)
            rob_pd.insert(loc=0, column="en_name", value=key_0)

            rob_pd_all = pd.concat([rob_pd_all, rob_pd])

    # save to file
    rob_pd_all.to_csv(output_file_csv, sep=',', index=False, float_format='%.2f')
