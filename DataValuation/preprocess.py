import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models import model_for_mnist
from dataloader import get_loaders_data
import math

# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Lambda, Add, AveragePooling2D
# from tensorflow.keras import regularizers
# from tensorflow.keras.optimizers import Adam,Adadelta
# from tensorflow.keras import layers, optimizers, datasets, Sequential, Model
# import tensorflow.keras.backend as K
#
# import matplotlib.pyplot as plt
#
# from tensorflow.keras.datasets import cifar10


def addGaussianNoise(x, scale=1, seed=12345):
    torch.random.manual_seed(seed)
    # a_norm = x.norm(2, dim=[-2, -1], keepdim=True)
    noise_x = x + torch.normal(0, scale, size=x.shape)
    # norm_x = noise_x/noise_x.norm(2, dim=[-2, -1], keepdim=True) * a_norm
    # no = torch.clip(torch.normal(0, scale, size=x.shape), -clip, clip)
    return torch.clip(noise_x, 0, 1)

class RandomizedResopnse:
    def __init__(self, p, d=10):
        self.d = d
        self.q = (1-p)/(d-1)  #p_ij
        self.p = p  #p_ii

    def __call__(self, y):
        pr = y * self.p + (1 - y) * self.q
        out = torch.multinomial(pr, num_samples=1)
        return out

def noise(data_owner, target_ind, scale):
    # if data_owner[0].dim() == 3:  # add noise on X
    for ind in target_ind:
        data_owner[ind] = addGaussianNoise(data_owner[ind], scale=scale)
    # else:  # add noise on y
    #     for ind in target_ind:
    #         data_owner[ind] = RandomRepsonse(data_owner[ind], epsilon=scale)
    return data_owner


def adv_fgsm(data, label, epsilon, attack='fgsm'):
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, requires_grad=False)
    data_loader = get_loaders_data((data, label), batch_size=100, transform=None)
    alpha = 1.25 * epsilon
    model = model_for_mnist().cuda()
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    lr_schedule = lambda t: np.interp([t], [0, 10 * 2 // 5, 10], [0, 5e-3, 0])[0]
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        for i, (X, y) in enumerate(data_loader):
            X, y = X.cuda(), y.cuda()
            lr = lr_schedule(epoch + (i + 1) / len(data_loader))
            opt.param_groups[0].update(lr=lr)

            if attack == 'fgsm':
                delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
                delta.requires_grad = True
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data = torch.max(torch.min(1 - X, delta.data), 0 - X)
                delta = delta.detach()
            # elif args.attack == 'none':
            #     delta = torch.zeros_like(X)
            elif attack == 'pgd':
                delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
                delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                for _ in range(20):
                    delta.requires_grad = True
                    output = model(X + delta)
                    loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                    grad = delta.grad.detach()
                    I = output.max(1)[1] == y
                    delta.data[I] = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)[I]
                    delta.data[I] = torch.max(torch.min(1-X, delta.data), 0-X)[I]
                delta = delta.detach()
            output = model(torch.clamp(X + delta, 0, 1))
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    data = data.unsqueeze(1).cuda()
    label = label.cuda()
    delta = torch.zeros_like(data).uniform_(-epsilon, epsilon).cuda()
    delta.requires_grad = True
    output = model(data + delta)
    loss = F.cross_entropy(output, label)
    loss.backward()
    grad = delta.grad.detach()
    delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
    delta.data = torch.max(torch.min(1 - data, delta.data), 0 - data)
    delta = delta.detach()
    adv_data = torch.clamp(data + delta, 0, 1)

    return adv_data.cpu().squeeze()


def clamp(x, lower_bound, upper_bound):
    return torch.max(torch.min(x, upper_bound), lower_bound)


def attack_pgd(model, X, y, epsilon, attack_iters, restarts=1):
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, requires_grad=False, dtype=torch.long)
    alpha = 1.25 * epsilon
    X, y = X.cuda(), y.cuda()
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
        delta.data = clamp(delta, 0 - X, 1 - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)[0]
            if len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            d = clamp(d, 0 - X, 1 - X)
            delta.data[index] = d[index]
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    # a_norm = X.cpu().norm(2)
    # noise_x = torch.clamp(X + max_delta, 0, 1).cpu()
    # norm_x = noise_x / noise_x.norm(2) * a_norm
    return torch.clamp(X + max_delta, 0, 1).cpu()
