import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

import numpy as np

from dataloader import MyDataset, get_loaders_data
from utils import Flatten
from preprocess import attack_pgd
# from transfer_learning import transfer_learning


def logistic_data_to_acc(x_train, y_train, val_data, kwargs={}):
    # if training data only consists of 1 class, the accuracy is 0.1
    x_val, y_val = val_data
    if not isinstance(y_val, torch.Tensor):
        y_val = torch.tensor(y_val, requires_grad=False, dtype=torch.float32)

    x_train = x_train.view((x_train.shape[0], -1))
    x_val = x_val.view((x_val.shape[0], -1))

    train_loader_1d = torch.utils.data.DataLoader(
        dataset=MyDataset(x_train, y_train, None),
        shuffle=True,
        batch_size=32,
    )

    test_loader_1d = torch.utils.data.DataLoader(
        dataset=MyDataset(x_val, y_val, None),
        shuffle=False,
        batch_size=len(y_val),
    )
    model = Small_models('logistic', input_size=x_train.shape[1])
    model.weight_reset()
    model.fit(train_loader_1d, epochs=10, lr=0.01)
    test_acc = model.val(test_loader_1d)
    return test_acc


def cnn_data_to_acc(x_train, y_train, val_data, kwargs={}):
    # if len(set(y_train.numpy())) == 1:
    #     return 0.1

    x_val, y_val = val_data
    if not isinstance(y_val, torch.Tensor):
        y_val = torch.tensor(y_val, requires_grad=False, dtype=torch.float32)

    y_train = y_train.squeeze()
    transform = None
    if transform is None:
        if x_train.dim() == 3:
            x_train = x_train.unsqueeze(1)
            x_val = x_val.unsqueeze(1)
            model = Small_models('cnn1')
        else:
            model = Small_models('cnn3')

    train_loader = torch.utils.data.DataLoader(
        dataset=MyDataset(x_train, y_train, transform),
        shuffle=True,
        batch_size=200,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=MyDataset(x_val, y_val, transform),
        shuffle=False,
        batch_size=len(y_val),
        pin_memory=True,
    )

    # proxy_model = Small_models('cnn')
    model.fit(train_loader, epochs=50, lr=0.001)

    test_acc = model.val(test_loader)

    return test_acc


def logistic_data_to_acc_tf(model_path, x_tf, y_tf, data_args):
    # if len(set(y_train.numpy())) == 1:
    #     return 0.1
    x_val, y_val = data_args
    y_tf = y_tf.squeeze()
    y_val = y_val.squeeze()

    transform = None
    tf_loader = get_loaders_data((x_tf, y_tf), 100, transform, 'logistic')
    val_loader = get_loaders_data((x_val, y_val), len(y_val), transform, 'logistic')

    model = Small_models('logistic')
    model.model.load_state_dict(torch.load(model_path))
    model.transfer(tf_loader, 10, 1e-1)
    test_acc = model.val(val_loader)
    return test_acc


# def mnist_cnn_data_to_acc_multiple(x_train, y_train, x_test, y_test, repeat=3, verbose=0):
#     acc_lst = []
#     for _ in range(repeat):
#         acc = mnist_cnn_data_to_acc(x_train, y_train, x_test, y_test, verbose)
#         acc_lst.append(acc)
#     return np.mean(acc_lst)


class Small_models:
    def __init__(self, model_name, input_size=784):
        torch.cuda.manual_seed(1)
        if model_name == 'cnn1':
            self.model = nn.Sequential(
                OrderedDict([
                    ('conv1', nn.Conv2d(1, 16, 4, stride=2, padding=1)),
                    ('relu1', nn.ReLU()),
                    ('conv2', nn.Conv2d(16, 32, 4, stride=2, padding=1)),
                    # ('dropout1', nn.Dropout2d(0.4)),
                    ('relu2', nn.ReLU()),
                    ('flatten', Flatten()),
                    ('linear1', nn.Linear(32 * 7 * 7, 512)),
                    # ('dropout2', nn.Dropout2d(0.4)),
                    ('relu3', nn.ReLU()),
                    ('linear2', nn.Linear(512, 10))
                ])).cuda()

            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        elif model_name == 'logistic':
            self.model = nn.Sequential(
                nn.Linear(input_size, 10)
            ).cuda()
        elif model_name == "cnn3":
            self.model = nn.Sequential(
                nn.Conv2d(3, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(),
                nn.Linear(120, 10)).cuda()

            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        else:
            raise ValueError('Unknown proxy_model')

    def weight_reset(self):
        if isinstance(self.model, nn.Conv2d) or isinstance(self.model, nn.Linear):
            self.model.reset_parameters()

    def fit(self, train_loader, epochs, lr):
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            train_acc = 0
            train_n = 0
            for i, (X, y) in enumerate(train_loader):
                X, y = X.cuda(), y.cuda()
                output = self.model(X)
                loss = criterion(output, y)
                loss.backward()
                opt.step()
                opt.zero_grad()

                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
            train_acc /= train_n
            # print('train_acc: {}'.format(train_acc))

    def val(self, val_loader):
        self.model.eval()
        test_acc = 0
        test_n = 0
        for i, (X, y) in enumerate(val_loader):
            X, y = X.cuda(), y.cuda()
            output = self.model(X)

            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)
        test_acc /= test_n
        return test_acc.__round__(4)

    def transfer(self, tf_loader, epochs, lr):
        # print('Transfer Learning')
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        # lr_scheduler = StepLR(opt, step_size=6, gamma=0.8)

        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            train_acc = 0
            train_n = 0
            for i, (X, y) in enumerate(tf_loader):
                X, y = X.cuda(), y.cuda()
                output = self.model(X)
                loss = criterion(output, y)
                loss.backward()
                opt.step()
                opt.zero_grad()

                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
            # lr_scheduler.step()
            train_acc /= train_n
            # print('transfer: {}'.format(train_acc))


# def evaluate_pgd(test_loader, proxy_model, attack_iters, restarts):
#     epsilon = 0.3
#     alpha = 1e-2
#     pgd_loss = 0
#     pgd_acc = 0
#     n = 0
#     proxy_model.eval()
#     for i, (X, y) in enumerate(test_loader):
#         X, y = X.cuda(), y.cuda()
#         pgd_delta = attack_pgd(proxy_model, X, y, epsilon, alpha, attack_iters, restarts)
#         # with torch.no_grad():
#         output = proxy_model(X + pgd_delta)
#         loss = F.cross_entropy(output, y)
#         pgd_loss += loss.item() * y.size(0)
#         pgd_acc += (output.max(1)[1] == y).sum().item()
#         n += y.size(0)
#     return pgd_acc / n
#
#
# def evaluate_standard(test_loader, proxy_model):
#     test_loss = 0
#     test_acc = 0
#     n = 0
#     proxy_model.eval()
#     # with torch.no_grad():
#     for i, (X, y) in enumerate(test_loader):
#         X, y = X.cuda(), y.cuda()
#         output = proxy_model(X)
#         loss = F.cross_entropy(output, y)
#         test_loss += loss.item() * y.size(0)
#         test_acc += (output.max(1)[1] == y).sum().item()
#         n += y.size(0)
#     return test_loss / n, test_acc / n
