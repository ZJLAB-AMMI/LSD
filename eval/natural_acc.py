import argparse
import logging
import os
import random
import sys
import time
import warnings

import pandas as pd

sys.path.append(os.getcwd())
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

import torch
from tqdm import tqdm

from utils import arguments
from utils import utils_models
from models.ensemble import Ensemble
from utils.utils_eval import get_root_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of natural_acc of Ensembles', add_help=True)
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
        model_num_list = [3, 5]
        alg_name_arr = ['baseline', 'adp', 'gal', 'dverge', 'trs', 'lcm_gal_4.0_0.5_4.0',
                        'lcm_gal_4.0_0.0_0.0', 'lcm_gal_4.0_1.0_0.0', 'lcm_gal_4.0_2.0_0.0', 'lcm_gal_4.0_4.0_0.0',
                        'lcm_gal_4.0_1.0_2.0', 'lcm_gal_4.0_2.0_2.0', 'lcm_gal_4.0_0.5_4.0', 'lcm_gal_4.0_2.0_4.0']
    elif args.dataset.lower() == 'fmnist':
        model_num_list = [3, 5]
        alg_name_arr = ['baseline', 'adp', 'gal', 'dverge', 'trs',
                        'lcm_gal_3.0_0.0_0.0', 'lcm_gal_3.0_1.0_0.0', 'lcm_gal_3.0_2.0_0.0', 'lcm_gal_3.0_4.0_0.0',
                        'lcm_gal_3.0_1.0_2.0', 'lcm_gal_3.0_1.0_4.0', 'lcm_gal_3.0_2.0_4.0', 'lcm_gal_3.0_4.0_4.0']
    else:
        model_num_list = [3, 5]
        alg_name_arr = ['baseline', 'adp', 'gal', 'dverge', 'trs',
                        'lcm_gal_3.0_0.0_0.0', 'lcm_gal_3.0_1.0_0.0', 'lcm_gal_3.0_2.0_0.0', 'lcm_gal_3.0_4.0_0.0',
                        'lcm_gal_3.0_1.0_2.0', 'lcm_gal_3.0_1.0_4.0', 'lcm_gal_3.0_2.0_4.0', 'lcm_gal_3.0_4.0_4.0', 'lcm_gal_3.0_2.0_1.0']

    random.seed(0)
    testloader = utils_models.get_testloader(args, batch_size=200, shuffle=False)

    output_root = os.path.join('./results', 'natural_acc', str(args.seed))
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    output_file_csv = os.path.join(output_root, '{}_natural_acc_results_{}_{}.csv'.format(args.dataset, 'clean', time.strftime('%Y_%m_%d_%H_%M')))

    rob = {}
    for _, alg_name in enumerate(alg_name_arr):
        rob[alg_name] = {}
        for _, model_num in enumerate(model_num_list):
            en_name = 'emodel_' + str(model_num)
            rob[alg_name][en_name] = {}

            for _, index in enumerate(range(0, model_num + 1)):
                net_name = 'Net_' + str(index)
                rob[alg_name][en_name][net_name] = 0.0

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
            models = utils_models.get_models(args, train=False, as_ensemble=False, model_file=args.model_file, leaky_relu=leaky_relu)
            ensemble = Ensemble(models)

            tqdm.write('model_num:{}'.format(model_num))
            with torch.no_grad():
                correct_arr = [0] * (len(models) + 1)
                total = 0
                for _, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.cuda(), targets.cuda()

                    for index, m in enumerate(models):
                        net_name = 'Net_' + str(index)
                        outputs = m(inputs)
                        _, predicted = outputs.max(1)
                        correct_arr[index] += predicted.eq(targets).sum().item()

                    outputs_ens = ensemble(inputs)
                    _, predicted_ens = outputs_ens.max(1)
                    correct_arr[-1] += predicted_ens.eq(targets).sum().item()
                    total += inputs.size(0)

                for _, index in enumerate(range(0, model_num + 1)):
                    net_name = 'Net_' + str(index)
                    rob[alg_name][en_name][net_name] = 100. * correct_arr[index] / total

    # dict to dict
    rob_csv = {}
    for k, model_num in enumerate(model_num_list):
        en_name = 'emodel_' + str(model_num)
        rob_csv[en_name] = {}

        for j, alg_name in enumerate(alg_name_arr):
            rob_csv[en_name][alg_name] = {}

            for _, index in enumerate(range(0, model_num + 1)):
                net_name = 'Net_' + str(index)
                rob_csv[en_name][alg_name][net_name] = rob[alg_name][en_name][net_name]

    # dict to pd
    rob_pd_all = None
    for key_0, value_0 in rob_csv.items():
        rob_pd = pd.DataFrame(data=value_0)
        rob_pd.insert(loc=0, column="model_num", value=key_0)

        rob_pd_all = pd.concat([rob_pd_all, rob_pd])
    print(rob_pd_all)

    # save to file
    rob_pd_all.to_csv(output_file_csv, sep=',', float_format='%.2f')
