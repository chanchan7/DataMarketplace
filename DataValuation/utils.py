import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentTypeError, Action

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import enum
import random


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def model_name(args, src, tgt):

    ext_n = str(args.model)
    ds_n = 'DeepSets'
    # if src is None:
    #     ext_n += '_' + str(src) + '2' + str(tgt)
    #     ds_n += '_' + str(src) + '2' + str(tgt)
    # else:
    #     ext_n += '_' + str(tgt)
    #     ds_n += '_' + str(tgt)
    ext_n += '_' + str(args.n_owners) + 'o' + str(args.data_size) + 's_' + str(12347)
    ds_n += '_' + str(args.n_owners) + 'o' + str(args.data_size) + 's_' + str(12347)
    if args.noisey:
        ext_n += '_' + str(args.prob_hold) + 'noisey_' + str(12347)
        ds_n += '_' + str(args.prob_hold) + 'noisey_' + str(12347)
    ext_n += '.pth'
    ds_n += '.pth'
    return ext_n, ds_n

# def addGaussianNoise(x_train, scale=1, seed=1):
#     torch.manual_seed(seed)
#     return torch.clip(x_train + torch.normal(0, scale, size=x_train.shape), 0, 1)


def collect_from_loader(dataloaders):
    src_loader, tgt_loader = dataloaders
    x, y = [], []
    if src_loader is None:
        for (image, label) in iter(tgt_loader):
            x.append(image)
            y.append(label)

    else:
        for (image, label) in iter(src_loader):
            x.append(image)
            y.append(label)
    data = (torch.cat(x), torch.cat(y))
    return data

def array_to_lst(X_feature):
    if type(X_feature) == list:
        return X_feature

    X_feature = list(X_feature)
    for i in range(len(X_feature)):
        X_feature[i] = X_feature[i].nonzero()[0]
    return X_feature


def featureExtract_cycda(x, ext):
    # x = x.cuda()
    # assert list(x.shape[1:]) == [1, ext.image_size, ext.image_size]
    score, out = ext(x, with_ft=True)
    return out.view(out.size(0), -1)


def featureExtract_cycday(x, y, ext):
    x = x.cuda()
    y = y.cuda()
    score, out = ext(x, y, with_ft=True)
    return out.view(out.size(0), -1)


def sample_count(n_select, n_owners, ub_prob):
    toss = np.random.uniform()
    # if toss > 1 - ub_prob:
    alpha = np.ones(n_owners) * 5
    # alpha[np.random.choice(range(n_owners))] = 3  #np.random.choice(range(5, 10))
    p = np.random.dirichlet(alpha=alpha)
    data_for_each_owner = np.random.multinomial(n_select, p, 1)[0]

    return data_for_each_owner


def val_to_dataind(v):
    # 利用二进制来选择所有的子集
    one_hot = np.array([int(x) for x in bin(v)[2:]])[::-1]
    return one_hot.nonzero()[0]


def dataind_to_val(arr):
    val = 0
    for i in arr:
        val += 2 ** i
    return val


class SiLU_(nn.Module):
    def forward(self, x):
        return 1 / 2 * x + 1 / 4 * x ** 2  # - 1 / 48 * x ** 4
        # return x ** 2


class Square(nn.Module):
    def forward(self, x):
        return x ** 2
        # return x ** 2


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def metric(accuracy, r_accuracy, percentage, low2high=True):
    if low2high:
        score = accuracy - r_accuracy
    else:
        score = r_accuracy - accuracy
    return np.sum(score * percentage)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def print_args(args):
    # message = [f'{name}: {colored_text(str(value), TermColors.FG.cyan)}' for name, value in vars(args).items()]
    message = [f'{name}: {str(value)}' for name, value in vars(args).items()]
    print(', '.join(message) + '\n')


def colored_text(msg, color):
    if isinstance(color, str):
        color = TermColors.FG.__dict__[color]
    return color.value + msg + TermColors.Control.reset.value


class TermColors:
    class Control(enum.Enum):
        reset = '\033[0m'
        bold = '\033[01m'
        disable = '\033[02m'
        underline = '\033[04m'
        reverse = '\033[07m'
        strikethrough = '\033[09m'
        invisible = '\033[08m'

    class FG(enum.Enum):
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class BG(enum.Enum):
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'

