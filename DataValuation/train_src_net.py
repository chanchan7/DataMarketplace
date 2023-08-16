from __future__ import print_function

import os
from os.path import join
import numpy as np
import argparse

# Import from torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models import get_model
from models.models import model_for_cifar10
from dataloader import get_loaders_data, get_loaders_name

# from test_task_net import test
from utils import SiLU_, Flatten


def train_epoch(loader, net, opt_net, flag=''):
    net.train()
    n = 0
    train_loss = 0
    train_corr = 0
    for batch_idx, (data, target) in enumerate(loader):
        data = data.cuda()
        target = target.cuda()
        opt_net.zero_grad()
        score = net(data)
        loss = net.criterion(score, target)
        loss.backward()
        opt_net.step()
        # if batch_idx % log_interval == 0:
        pred = score.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum()

        n += len(pred)
        train_loss += loss.item()
        train_corr += correct.item()

    avg_loss = train_loss / n
    acc = train_corr / n * 100.0
    return {flag + 'Train/Acc': '{:.4f}'.format(acc)}  ## flag+'Train/Loss': round(avg_loss, 3),


def test_epoch(loader, net):
    net.eval()
    n = 0
    test_loss = 0
    test_corr = 0
    for batch_idx, (data, target) in enumerate(loader):
        data = data.cuda()
        target = target.cuda()
        score = net(data)
        loss = net.criterion(score, target)
        pred = score.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum()

        n += len(pred)
        test_loss += loss.item()
        test_corr += correct.item()

    avg_loss = test_loss / n
    acc = test_corr / n * 100.0
    return {'Test/Loss': '{:.4f}'.format(avg_loss), 'Test/Acc': '{:.4f}'.format(acc)}


def train_gan_epoch(src_loader, tgt_loader, net, opt_dis):
    joint_loader = zip(src_loader, tgt_loader)

    n = 0
    acc = 0
    net.train()
    for batch_idx, ((data_s, _), (data_t, _)) in enumerate(joint_loader):
        data_s, data_t = data_s.cuda(), data_t.cuda()

        opt_dis.zero_grad()
        # extract and concat features
        score_s = net.src_net(data_s, with_ft=False)
        score_t = net.tgt_net(data_t, with_ft=False)
        f = torch.cat((score_s, score_t), 0)
        pred_concat = net.discriminator(f)

        # prepare real and fake labels: source=1, target=0
        target_dom_s = torch.ones(len(data_s), requires_grad=False, dtype=torch.long)
        target_dom_t = torch.zeros(len(data_t), requires_grad=False, dtype=torch.long)
        label_concat = torch.cat((target_dom_s, target_dom_t), 0).cuda()

        # compute loss for disciminator
        loss_dis = net.gan_criterion(pred_concat, label_concat)
        loss_dis.backward()
        opt_dis.step()
        pred_dis = torch.squeeze(pred_concat.max(1)[1])
        correct = pred_dis.eq(label_concat).cpu().sum()

        n += len(pred_dis)
        acc += correct.item()
    acc = acc / n
    return acc, {'[DisTrain] Train/Acc': '{:.4f}'.format(acc)}


def train_tgt(tgt_loader, net, opt_tgt):
    net.train()

    n = 0
    train_loss = 0
    for batch_idx, (data_t, _) in enumerate(tgt_loader):
        data_t = data_t.cuda()

        opt_tgt.zero_grad()

        score_t = net.tgt_net(data_t, with_ft=False)
        pred_tgt = net.discriminator(score_t)
        label_tgt = torch.ones(pred_tgt.size(0), requires_grad=False, dtype=torch.long).cuda()
        loss_gan_t = net.gan_criterion(pred_tgt, label_tgt)
        loss_gan_t.backward()
        opt_tgt.step()

        n += len(label_tgt)
        train_loss += loss_gan_t.item()
    avg_loss = train_loss / n
    return {'[TgtTrain] Train/Loss': round(avg_loss, 4)}


def pretraining_from_data(train_data, model, num_cls, outfile, num_epoch, batch, lr):
    """Train a classification net."""
    net = get_model(model, num_cls=num_cls).cuda()

    print('-------Training net--------')
    print(net)

    train_data = get_loaders_data(train_data, batch, transform=None)

    opt_net = optim.Adam(net.parameters(), lr=lr)

    # print('Training {} model for {}'.format(model, data_name))
    for epoch in range(num_epoch):
        metric = train_epoch(train_data, net, opt_net, flag='Pre')
        print(metric)

    print('Saving to', outfile)
    net.save(outfile)

    return net


def train_source_from_loader(data_name, data_loader, model, num_cls, outfile, num_epoch=100, lr=1e-4):
    """Train a classification net."""

    ############
    # Load Net #
    ############
    net = get_model(model, num_cls=num_cls)
    print('-------Training net--------')
    print(net)

    ###################
    # Setup Optimizer #
    ###################
    opt_net = optim.Adam(net.parameters(), lr=lr)

    #########
    # Train #
    #########
    print('Training {} model for {}'.format(model, data_name))
    for epoch in range(num_epoch):
        metric = train_epoch(data_loader, net, opt_net, flag='SrcTrain')
        print(metric)

    ############
    # Save net #
    ############
    print('Saving to', outfile)
    net.save(outfile)

    return net

