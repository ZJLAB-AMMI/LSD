import os
import pickle
import random
import sys

import torch
from sklearn.manifold import TSNE

from utils import arguments
from utils import utils_models
from utils.utils_eval import get_root_path

sys.path.append(os.getcwd())
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

import argparse
import logging
import os
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)


def pick_out_correctly_classified_samples(args, models, testloader):
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
    subset_loader = utils_models.get_testloader(args, batch_size=200, shuffle=False, subset_idx=correct_idx)
    return subset_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of Transferability within Ensembles', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.transf_eval_args(parser)
    arguments.dverge_train_args(parser)
    arguments.lcm_gal_train_args(parser)
    args = parser.parse_args()

    for dataset in ['cifar10', 'mnist', 'fmnist']:
        random.seed(0)
        torch.manual_seed(0)

        args.__setattr__('dataset', dataset)
        if args.dataset.lower() == 'cifar10':
            alg_name_arr = ['baseline', 'lcm_gal_4.0_0.5_4.0', 'adp', 'gal', 'dverge', 'trs']  # ['baseline', 'lcm_gal_4.0_0.5_4.0']
            args.__setattr__('arch', 'ResNet')
            args.__setattr__('depth', 20)
            args.__setattr__('epochs', 200)
        elif args.dataset.lower() == 'fmnist':
            alg_name_arr = ['baseline', 'lcm_gal_3.0_4.0_4.0', 'adp', 'gal', 'dverge', 'trs']  # ['baseline', 'lcm_gal_3.0_4.0_4.0']
            args.__setattr__('arch', 'LeNet')
            args.__setattr__('depth', 5)
            args.__setattr__('epochs', 20)
        else:
            alg_name_arr = ['baseline', 'lcm_gal_3.0_2.0_1.0', 'adp', 'gal', 'dverge', 'trs']  # ['baseline', 'lcm_gal_3.0_2.0_1.0']
            args.__setattr__('arch', 'LeNet')
            args.__setattr__('depth', 5)
            args.__setattr__('epochs', 20)

        # get data loaders
        input_file = os.path.join(os.path.join(args.data_dir, 'pick_out_correctly_idx'), 'idx_evaluation_{}.pkl'.format(args.dataset))
        with open(input_file, "rb") as tf:
            subset_correctly_idx = pickle.load(tf)
        testloader = utils_models.get_testloader(args, batch_size=200, shuffle=False, subset_idx=subset_correctly_idx)

        plt.figure(num=0, figsize=(3 * 4, len(alg_name_arr) * 4), dpi=120)  # 设置图形 len(alg_name_arr), len(models)
        for row_num, alg_name in enumerate(alg_name_arr):
            args.__setattr__('model_file', os.path.join(get_root_path(args, alg_name), 'epoch_%d.pth' % args.epochs))

            pick_out_samples = True
            leaky_relu = True if 'gal' in args.model_file else False
            ensemble = utils_models.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=leaky_relu)
            models = ensemble.models

            subset_loader = pick_out_correctly_classified_samples(args, models, testloader)
            for j, m in enumerate(models):
                input_vec_s = []
                labels = []
                with torch.no_grad():
                    for index, (x, y) in enumerate(subset_loader if pick_out_samples == True else testloader):
                        x, y = x.cuda(), y.cuda()
                        input_vec_s.append(m.get_predicted_label(x) + 1e-6)
                        labels.append(y)

                input_vec_s = torch.cat(input_vec_s, dim=0)
                input_vec_s = input_vec_s.cuda()
                print(input_vec_s.shape)

                with torch.no_grad():
                    print('Begining dataset:{}\t alg_name: {}\t models: {}'.format(args.dataset.lower(), alg_name, j))
                    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
                    result_2D = tsne_2D.fit_transform(input_vec_s.cpu())

                labels = torch.cat(labels, dim=0)
                labels = labels.tolist()
                plt.subplot(len(alg_name_arr), len(models), row_num * len(models) + j + 1)
                plt.title('{}, {}, model_{}'.format(args.dataset.lower(), alg_name, j))
                plt.scatter(result_2D[:, 0], result_2D[:, 1], 2, labels)
        # plt.title('TSNE {}'.format(args.dataset.lower()))
        plt.savefig('./figure/TSNE_{}.jpg'.format(args.dataset.lower()))
        plt.show()
        print('{} Finished'.format(args.dataset.lower()))
