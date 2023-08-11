"""
Train GAN and utility dunctions (Deepsets models)
"""

from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
import os
from models.deepsets_net import DeepSets
from deepsets_training import utility_deepset
from sample_utility import sample_utility, sample_utility_veryub, utility_ds, generate_deepsets
from proxy_models import logistic_data_to_acc
from models.models import get_model
from train_uf import train as train_ds


def train_gan(src_loader, tgt_loader, net, opt_tgt, opt_dis, epoch, logger, verbose=True, with_ft=False):
    N = max(len(src_loader), len(tgt_loader))
    joint_loader = zip(src_loader, cycle(tgt_loader))
    # log_interval = N  # specifies how often to display

    net.train()

    last_update = -1
    for batch_idx, ((data_s, _), (data_t, _)) in enumerate(joint_loader):
        data_s, data_t = data_s.cuda(), data_t.cuda()
        # log basic adda train info
        info_str = "[Train Adda] Epoch:{}".format(epoch)

        opt_dis.zero_grad()
        # extract and concat features
        score_s = net.src_net(data_s, with_ft=with_ft)
        score_t = net.tgt_net(data_t, with_ft=with_ft)
        if with_ft:
            f = torch.cat((score_s[1], score_t[1]), 0)
        else:
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
        acc = (pred_dis == label_concat).float().mean()

        info_str += " Dis/Acc: {:.3f}, Dis/Loss: {:.3f}".format(acc.item() * 100, loss_dis.item())

        ###########################
        # Optimize target network #
        ###########################

        # only update net if discriminator is strong
        if acc.item() > 0.6:
            last_update = batch_idx
            # for i in range(15):
            opt_dis.zero_grad()
            opt_tgt.zero_grad()

            score_t = net.tgt_net(data_t, with_ft=with_ft)
            if with_ft:
                pred_tgt = net.discriminator(score_t[1])
            else:
                pred_tgt = net.discriminator(score_t)
            label_tgt = torch.ones(pred_tgt.size(0), requires_grad=False, dtype=torch.long).cuda()
            loss_gan_t = net.gan_criterion(pred_tgt, label_tgt)
            loss_gan_t.backward()
            opt_tgt.step()
            pred_tgt = torch.squeeze(pred_tgt.max(1)[1])
            tgt_acc = (pred_tgt == label_tgt).float().mean()

            # log net update info
            info_str += ", Tgt/Acc: {:.3f}, Gan/Loss: {:.3f}".format(tgt_acc.item() * 100, loss_gan_t.item())

        # if batch_idx % log_interval == 0:
    if verbose:
        logger.info(info_str)

    # return last_update


def train_gan_weight(src_loader, tgt_loader, net, opt_tgt, opt_dis, epoch, logger, verbose=True):
    N = max(len(src_loader), len(tgt_loader))
    joint_loader = zip(src_loader, tgt_loader)
    # log_interval = N  # specifies how often to display

    net.train()

    last_update = -1
    for batch_idx, ((data_s, _), (data_t, _, weight)) in enumerate(joint_loader):
        data_s, data_t = data_s.cuda(), data_t.cuda()
        weight = weight.cuda()
        # log basic adda train info
        info_str = "[Train Adda] Epoch:{}".format(epoch)

        opt_dis.zero_grad()
        # extract and concat features
        score_s = net.src_net(data_s)
        score_t = net.tgt_net(data_t)  # NOTE!!!!!!!! score is not feature (with_ft=True)
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
        acc = (pred_dis == label_concat).float().mean()

        info_str += " Dis/Acc: {:.3f}, Dis/Loss: {:.3f}".format(acc.item() * 100, loss_dis.item())

        ###########################
        # Optimize target network #
        ###########################

        # only update net if discriminator is strong
        if acc.item() > 0.6:
            last_update = batch_idx
            # for i in range(15):
            opt_dis.zero_grad()
            opt_tgt.zero_grad()

            score_t = net.tgt_net(data_t)
            pred_tgt = net.discriminator(score_t)
            label_tgt = torch.ones(pred_tgt.size(0), requires_grad=False, dtype=torch.long).cuda()
            loss_gan_t = F.cross_entropy(pred_tgt, label_tgt, reduction='none')
            loss_gan_t = (loss_gan_t*weight).sum()
            loss_gan_t.backward()
            opt_tgt.step()

            # log net update info
            info_str += ", Gan/Loss: {:.3f}".format(loss_gan_t.item())

        # if batch_idx % log_interval == 0:
    if verbose:
        logger.info(info_str)

    return last_update


def train_adda_uf(src, tgt, sample_loaders, utility_data, train_set, test_set, num_cls,
                  epochs, pretrain_net_file, out_dir, logger,  model, ext_file, ds_file, lr_adda, lr_deepsets, device):
    src_loader, tgt_loader = sample_loaders
    extractor = get_model('AddaNet', model=model, num_cls=num_cls, discrim_feat=False, src_weights_init=pretrain_net_file).to(device)
    extractor.train()

    deepset = get_model('DeepSets', in_features=extractor.src_net.output_dim, hidden_ext=512, hidden_reg=512).to(device)
    ds_model = utility_deepset(model=deepset, lr=lr_deepsets, device=device)

    logger.info('Training Adda {} model for {}->{}'.format(model, src, tgt))

    ## -------------------------------------------Optimizer----------------------------------------------
    opt_src = optim.Adam(extractor.src_net.parameters(), lr=lr_adda['src'])
    opt_tgt = optim.Adam(extractor.tgt_net.parameters(), lr=lr_adda['tgt'])
    opt_dis = optim.Adam(extractor.discriminator.parameters(), lr=lr_adda['dis'])

    ## -------------------------------------------Train adda----------------------------------------------
    for epoch in range(epochs):
        train_gan(src_loader, tgt_loader, extractor, opt_tgt, opt_dis, epoch, logger, with_ft=False)
        if (epoch + 1) % 10 == 0:
            if utility_data.dim() != 4:
                utility_data = utility_data.unsqueeze(1)
            train_loss, test_loss = train_ds(utility_data, train_set, test_set, ds_model,
                                             extractor.src_net, opt_src, 1, logger)
            logger.info('Epoch: %s, Train Loss: %s, Test Loss: %s' % (epoch + 1, train_loss, test_loss))


    # train_ds(utility_data, train_set, test_set, ds_model, extractor.src_net, opt_src, 1, logger)

    ## ----------------------------------------Save proxy_model-------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    # logger.info('Extractor model saving to: {}'.format(ext_file))
    # extractor.tgt_net.save(ext_file)
    #
    # logger.info("Deepsets model saving to: {}".format(ds_file))
    # ds_model.model.save(ds_file)

    return extractor.tgt_net, ds_model.model
