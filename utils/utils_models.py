import random

import torch
import torch.nn as nn
import torch.optim as optim
from advertorch.utils import NormalizeByChannelMeanStd
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.ensemble import Ensemble
from models.lenet import LeNet
from models.resnet import ResNet


def get_models(args, train=True, as_ensemble=False, model_file=None, leaky_relu=False):
    models = []

    if args.dataset == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).cuda()
        std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()
        normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)
    elif args.dataset == 'mnist':
        mean = torch.tensor([0.1307], dtype=torch.float32).cuda()
        std = torch.tensor([0.3081], dtype=torch.float32).cuda()
        normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)
    else:
        mean = torch.tensor([0.2854], dtype=torch.float32).cuda()
        std = torch.tensor([0.3527], dtype=torch.float32).cuda()
        normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)

    if model_file:
        state_dict = torch.load(model_file)
        if train:
            print('Loading pre-trained models...')

    iter_m = state_dict.keys() if model_file else range(args.model_num)

    for i in iter_m:
        if args.arch.lower() == 'resnet':
            model = ResNet(depth=args.depth, num_classes=args.num_classes, leaky_relu=leaky_relu)
        elif args.arch.lower() == 'lenet':
            model = LeNet()
        else:
            raise ValueError('[{:s}] architecture is not supported yet...')
        # we include input normalization as a part of the model
        model = ModelWrapper(model, normalizer)
        if model_file:
            model.load_state_dict(state_dict[i])
        if train:
            model.train()
        else:
            model.eval()
        model = model.cuda()
        models.append(model)

    if as_ensemble:
        assert not train, 'Must be in eval mode when getting models to form an ensemble'
        ensemble = Ensemble(models)
        ensemble.eval()
        return ensemble
    else:
        return models


def get_ensemble(args, train=False, model_file=None, leaky_relu=False):
    return get_models(args, train, as_ensemble=True, model_file=model_file, leaky_relu=leaky_relu)


class ModelWrapper(nn.Module):
    def __init__(self, model, normalizer):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.normalizer = normalizer

    def forward(self, x):
        x = self.normalizer(x)
        return self.model(x)

    def get_features(self, x, layer, before_relu=True):
        x = self.normalizer(x)
        return self.model.get_features(x, layer, before_relu)

    def get_input_vec(self, x):
        x = self.normalizer(x)
        return self.model.get_input_vec(x)

    def get_predicted_label(self, x):
        x = self.normalizer(x)
        predicted_label = self.model.get_predicted_label(x)
        return predicted_label


def get_loaders(args, add_gaussian=False):
    kwargs = {'num_workers': 4, 'batch_size': args.batch_size, 'shuffle': True, 'pin_memory': False}  # pin_memory= True

    if args.dataset == 'fmnist':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), ])

        trainset = datasets.FashionMNIST(root=args.data_dir, train=True, transform=transform_train, download=True)
        testset = datasets.FashionMNIST(root=args.data_dir, train=False, transform=transform_test, download=True)
    elif args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), ])

        trainset = datasets.MNIST(root=args.data_dir, train=True, transform=transform_train, download=True)
        testset = datasets.MNIST(root=args.data_dir, train=False, transform=transform_test, download=True)
    else:
        if not add_gaussian:
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        else:
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), AddGaussianNoise(0., 0.045)])
        transform_test = transforms.Compose([transforms.ToTensor(), ])

        trainset = datasets.CIFAR10(root=args.data_dir, train=True, transform=transform_train, download=True)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_test, download=True)

    trainloader = DataLoader(trainset, **kwargs)
    testloader = DataLoader(testset, num_workers=4, batch_size=100, shuffle=False, pin_memory=False)  # pin_memory= True
    return trainloader, testloader


def get_testloader(args, train=False, batch_size=100, shuffle=False, subset_idx=None):
    kwargs = {'num_workers': 4, 'batch_size': batch_size, 'shuffle': shuffle, 'pin_memory': False}
    transform_test = transforms.Compose([transforms.ToTensor(), ])

    if args.dataset == 'fmnist':
        if subset_idx is not None:
            testset = Subset(datasets.FashionMNIST(root=args.data_dir, train=train, transform=transform_test, download=False), subset_idx)
        else:
            testset = datasets.FashionMNIST(root=args.data_dir, train=train, transform=transform_test, download=False)
    elif args.dataset == 'mnist':
        if subset_idx is not None:
            testset = Subset(datasets.MNIST(root=args.data_dir, train=train, transform=transform_test, download=False), subset_idx)
        else:
            testset = datasets.MNIST(root=args.data_dir, train=train, transform=transform_test, download=False)
    else:
        if subset_idx is not None:
            testset = Subset(datasets.CIFAR10(root=args.data_dir, train=train, transform=transform_test, download=False), subset_idx)
        else:
            testset = datasets.CIFAR10(root=args.data_dir, train=train, transform=transform_test, download=False)

    testloader = DataLoader(testset, **kwargs)
    return testloader


class DistillationLoader:
    def __init__(self, seed, target):
        self.seed = iter(seed)
        self.target = iter(target)

    def __len__(self):
        return len(self.seed)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            si, sl = next(self.seed)
            ti, tl = next(self.target)
            return si, sl, ti, tl
        except StopIteration as e:
            raise StopIteration


def get_optimizers(args, models):
    optimizers = []
    for model in models:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-7)
        optimizers.append(optimizer)
    return optimizers


def get_schedulers(args, optimizers):
    schedulers = []
    for optimizer in optimizers:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.sch_intervals, gamma=args.lr_gamma)
        schedulers.append(scheduler)
    return schedulers


# This is used for training of GAL
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., prob=.5):
        self.std = std
        self.mean = mean
        self.prob = prob

    def __call__(self, tensor):
        if random.random() > self.prob:
            return tensor
        else:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
