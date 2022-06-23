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
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import arguments
from utils import utils_models
from utils.utils_eval import get_adversary_obj, get_root_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of Transferability within Ensembles', add_help=True)

    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.dverge_train_args(parser)
    arguments.lcm_gal_train_args(parser)
    arguments.transf_eval_args(parser)
    args = parser.parse_args()
    args.__setattr__('start_from', 'scratch')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.dataset.lower() == 'cifar10':
        eps_list = [0.04]
        model_num_list = [3]
        alg_name_arr = ['baseline', 'adp', 'gal', 'dverge', 'trs',
                        'lcm_gal_4.0_0.0_0.0', 'lcm_gal_4.0_1.0_0.0', 'lcm_gal_4.0_2.0_0.0', 'lcm_gal_4.0_4.0_0.0',
                        'lcm_gal_4.0_1.0_2.0', 'lcm_gal_4.0_2.0_2.0', 'lcm_gal_4.0_0.5_4.0', 'lcm_gal_4.0_2.0_4.0']
    elif args.dataset.lower() == 'fmnist':
        eps_list = [0.2]
        model_num_list = [3]
        alg_name_arr = ['baseline', 'adp', 'gal', 'dverge', 'trs',
                        'lcm_gal_3.0_0.0_0.0', 'lcm_gal_3.0_1.0_0.0', 'lcm_gal_3.0_2.0_0.0', 'lcm_gal_3.0_4.0_0.0',
                        'lcm_gal_3.0_1.0_2.0', 'lcm_gal_3.0_1.0_4.0', 'lcm_gal_3.0_2.0_4.0', 'lcm_gal_3.0_4.0_4.0']
    else:
        eps_list = [0.25]
        model_num_list = [3]
        alg_name_arr = ['baseline', 'adp', 'gal', 'dverge', 'trs',
                        'lcm_gal_3.0_0.0_0.0', 'lcm_gal_3.0_1.0_0.0', 'lcm_gal_3.0_2.0_0.0', 'lcm_gal_3.0_4.0_0.0',
                        'lcm_gal_3.0_1.0_2.0', 'lcm_gal_3.0_1.0_4.0', 'lcm_gal_3.0_2.0_4.0', 'lcm_gal_3.0_4.0_4.0', 'lcm_gal_3.0_2.0_1.0']

    attack_list = ['bim', 'fgsm', 'mim', 'pgd']

    random.seed(0)
    input_file = os.path.join(os.path.join(args.data_dir, 'pick_out_correctly_idx'), 'idx_evaluation_{}.pkl'.format(args.dataset))
    with open(input_file, "rb") as tf:
        subset_correctly_idx = pickle.load(tf)
    testloader = utils_models.get_testloader(args, batch_size=200, shuffle=False, subset_idx=subset_correctly_idx)

    output_root = os.path.join('./results', 'transferability', str(args.seed))
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    output_file_csv = os.path.join(output_root, '{}_transferability_results_{}_{}.csv'.format(args.dataset, 'clean', time.strftime('%Y_%m_%d_%H_%M')))

    loss_fn = nn.CrossEntropyLoss()

    # dict init
    rob = {}
    for attack_type in attack_list:
        rob[attack_type] = {}
        for alg_name in alg_name_arr:
            rob[attack_type][alg_name] = {}

            for model_num in model_num_list:
                en_name = 'emodel_' + str(model_num)
                rob[attack_type][alg_name][en_name] = {}

                for eps in eps_list:
                    eps_name = 'eps_' + str(int(eps * 255))
                    rob[attack_type][alg_name][en_name][eps_name] = {}

                    for _, index_i in enumerate(range(0, model_num)):
                        net_name_i = 'Net_' + str(index_i)
                        rob[attack_type][alg_name][en_name][eps_name][net_name_i] = {}

                        for _, index_j in enumerate(range(0, model_num + 1)):
                            net_name_j = 'Net_' + str(index_j)
                            rob[attack_type][alg_name][en_name][eps_name][net_name_i][net_name_j] = 0

    for attack_type in attack_list:
        tqdm.write('attack_type:{}'.format(attack_type))
        for alg_name in alg_name_arr:
            alg_name_t = alg_name

            tqdm.write('alg_name:{}'.format(alg_name))
            for model_num in model_num_list:
                en_name = 'emodel_' + str(model_num)
                args.__setattr__('model_num', model_num)

                save_root = get_root_path(args, alg_name_t)
                args.__setattr__('model_file', os.path.join(save_root, 'epoch_%d.pth' % args.epochs))

                leaky_relu = True if 'gal' in args.model_file else False
                ensemble = utils_models.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=leaky_relu)
                models = ensemble.models

                # pick out samples that are correctly classified by all submodels
                correct = []
                for m in models:
                    correct_m = []
                    for (x, y) in testloader:
                        x, y = x.cuda(), y.cuda()

                        outputs = m(x)
                        _, pred = outputs.max(1)
                        correct_m.append(pred.eq(y))
                    correct_m = torch.cat(correct_m)
                    correct.append(correct_m)
                correct = torch.stack(correct, dim=-1).all(-1)
                correct_idx = correct.nonzero().squeeze(-1)

                random.seed(0)
                subset_idx = correct_idx[random.sample(range(correct_idx.size(0)), args.subset_num)].cpu()
                subset_loader = utils_models.get_testloader(args, batch_size=200, shuffle=False, subset_idx=torch.tensor(subset_correctly_idx)[subset_idx].tolist())

                tqdm.write('model_num:{}'.format(model_num))
                for eps in eps_list:
                    eps_name = 'eps_' + str(int(eps * 255))

                    correct_or_not_rs = torch.zeros((len(models), len(models) + 1, args.subset_num, args.random_start), dtype=torch.bool)
                    for rs in range(args.random_start):
                        random.seed(rs)
                        torch.manual_seed(rs)
                        total = 0
                        for (x, y) in subset_loader:
                            x, y = x.cuda(), y.cuda()

                            adv_list = []
                            for i, m in enumerate(models):
                                adversary = get_adversary_obj(args, m, loss_fn, eps, attack_type)
                                adv = adversary.perturb(x, y)
                                adv_list.append(adv)

                            for i, adv in enumerate(adv_list):
                                for j, m in enumerate(models):
                                    if j == i:
                                        outputs = m(x)
                                        _, pred = outputs.max(1)
                                        assert pred.eq(y).all()

                                    outputs = m(adv)
                                    _, pred = outputs.max(1)
                                    correct_or_not_rs[i, j, total:total + x.size(0), rs] = pred.eq(y)

                                outputs = ensemble(adv)
                                _, pred = outputs.max(1)
                                correct_or_not_rs[i, len(models), total:total + x.size(0), rs] = pred.eq(y)

                            total += x.size(0)

                    correct_or_not_rs = torch.all(correct_or_not_rs, dim=-1)
                    asr = np.zeros((len(models), len(models) + 1))

                    for _, index_i in enumerate(range(0, len(models))):
                        net_name_i = 'Net_' + str(index_i)
                        for _, index_j in enumerate(range(0, len(models) + 1)):
                            net_name_j = 'Net_' + str(index_j)
                            rob[attack_type][alg_name][en_name][eps_name][net_name_i][net_name_j] = 1 - correct_or_not_rs[index_i, index_j, :].sum().item() / args.subset_num

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
                    rob_csv[en_name][attack_type][eps_name][alg_name] = {}
                    for _, index_i in enumerate(range(0, model_num)):
                        net_name_i = 'Net_' + str(index_i)
                        rob_csv[en_name][attack_type][eps_name][alg_name][net_name_i] = {}

                        for _, index_j in enumerate(range(0, model_num + 1)):
                            net_name_j = 'Net_' + str(index_j)
                            rob_csv[en_name][attack_type][eps_name][alg_name][net_name_i][net_name_j] = rob[attack_type][alg_name][en_name][eps_name][net_name_i][net_name_j]

    # dict to pd
    rob_pd_all = None
    for key_0, value_0 in rob_csv.items():
        rob_pd_all_1 = None
        for key_1, value_1 in value_0.items():
            for key_2, value_2 in value_1.items():
                rob_pd_all_2 = None
                for key_3, value_3 in value_2.items():
                    rob_pd = pd.DataFrame(data=value_3).T
                    for i in range(1, 6 - len(value_3)):
                        rob_pd['Net_' + str(i + len(value_3))] = ''
                    rob_pd_all_2 = pd.concat([rob_pd_all_2, rob_pd], axis=1)

                rob_pd_all_2.insert(loc=0, column="eps", value=int(key_2.split('_')[-1]))
                rob_pd_all_2.insert(loc=0, column="attack_type", value=key_1)
                rob_pd_all_2.insert(loc=0, column="en_name", value=key_0)
                rob_pd_all_1 = pd.concat([rob_pd_all_1, rob_pd_all_2], axis=0)
        rob_pd_all = pd.concat([rob_pd_all, rob_pd_all_1], axis=0)
    print(rob_pd_all)

    # save to file
    rob_pd_all.to_csv(output_file_csv, sep=',', index=False, float_format='%.2f')
