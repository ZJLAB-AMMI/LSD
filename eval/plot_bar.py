import argparse
import logging
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot_bar', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.transf_eval_args(parser)
    arguments.dverge_train_args(parser)
    arguments.lcm_gal_train_args(parser)
    args = parser.parse_args()

    alg_name = 'baseline'
    save_root = get_root_path(args, alg_name)
    args.__setattr__('model_file', os.path.join(save_root, 'epoch_%d.pth' % args.epochs))
    if 'gal' in args.model_file:
        leaky_relu = True
    else:
        leaky_relu = False
    ensemble_base = utils_models.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=leaky_relu)
    models_base = ensemble_base.models

    alg_name = 'lcm_gal'
    save_root = get_root_path(args, alg_name)
    args.__setattr__('model_file', os.path.join(save_root, 'epoch_%d.pth' % args.epochs))
    if 'gal' in args.model_file:
        leaky_relu = True
    else:
        leaky_relu = False
    ensemble = utils_models.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=leaky_relu)
    models = ensemble.models

    random.seed(0)
    input_file = os.path.join(os.path.join(args.data_dir, 'pick_out_correctly_idx'), 'idx_evaluation_{}.pkl'.format(args.dataset))
    with open(input_file, "rb") as tf:
        subset_correctly_idx = pickle.load(tf)

    subset_idx = random.sample(subset_correctly_idx.tolist(), 200)
    testloader = utils_models.get_testloader(args, batch_size=200, shuffle=False, subset_idx=subset_idx)

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
    correct_idx = correct.nonzero().squeeze(-1).cpu()
    subset_loader = utils_models.get_testloader(args, batch_size=1, shuffle=True, subset_idx=correct_idx)


    def pred_probs_bar_show(inputs, pred_probs, index, models_cnt, alg_name):

        plt.figure(num=index, figsize=(16, 9), dpi=120)  # 设置图形
        plt.subplot(1, models_cnt + 2, 1)
        inputs = torch.squeeze(inputs, 0).cpu()
        pic = inputs.reshape(3, 32, 32)
        pic = np.transpose(pic, (1, 2, 0))

        plt.imshow(pic)

        for i, item in enumerate(pred_probs):
            plt.subplot(1, models_cnt + 2, i + 2)
            plt.bar(range(len(item)), item)
            font = {'weight': 'normal', 'size': 20}  # bold,normal 'family': 'Times New Roman',

            if i == 0:
                plt.xlabel('y_true', font)
            else:
                plt.xlabel('Prediction {}'.format(i), font)

            plt.xticks(np.arange(10), ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], rotation=45, fontsize=10)

        if alg_name == 'lcm':
            plt.savefig('./figure/{}_figure_{}_{}_{}.jpg'.format(alg_name, args.ld_coeff, models_cnt, index))
        else:
            plt.savefig('./figure/{}_figure_{}_{}.jpg'.format(alg_name, models_cnt, index))
        plt.show()


    with torch.no_grad():
        for index, (inputs, targets) in enumerate(subset_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            y_true = torch.zeros(inputs.size(0), 10).cuda()
            y_true.scatter_(1, targets.view(-1, 1), 1)

            pred_probs = []
            pred_probs.append(y_true)
            for j, m in enumerate(models):
                pred_probs.append(F.softmax(m(inputs), dim=-1))

            pred_probs = torch.cat(pred_probs, dim=0)
            pred_probs = pred_probs.cpu().numpy()
            print(pred_probs)
            pred_probs_bar_show(inputs, pred_probs, index, len(models), 'lcm_gal')

            pred_probs = []
            pred_probs.append(y_true)
            for j, m in enumerate(models_base):
                pred_probs.append(F.softmax(m(inputs), dim=-1))

            pred_probs = torch.cat(pred_probs, dim=0)
            pred_probs = pred_probs.cpu().numpy()
            print(pred_probs)
            pred_probs_bar_show(inputs, pred_probs, index, len(models_base), 'baseline')

            if index >= 4:
                break
