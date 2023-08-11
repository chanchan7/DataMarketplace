"""Experiments marignal contribution of dataset"""

import sys
sys.path.append('..')
import torch
import numpy as np
import copy
import logging
import os
from csv import writer

from dataloader import construct_dataowner, sample_from_dataowner
from proxy_models import cnn_data_to_acc, logistic_data_to_acc
from preprocess import addGaussianNoise, RandomizedResopnse
from experimental_setting import experimental_setting
from train_src_net import train_epoch, test_epoch, train_gan_epoch, train_tgt, pretraining_from_data
from train_adda import train_gan, train_gan_weight
from models.models import get_model
from models.models import model_for_mnist, model_for_cifar10, Classifier, Integrater
from dataloader import MyDataset, get_loaders_name, MyDataset3
from torch.optim.lr_scheduler import StepLR
from utils import collect_from_loader

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        # filename=logfile,
    )
    logger.info(args)

    rng = np.random.default_rng(args.seed)
    data_owner_X, data_owner_y, valid_data = construct_dataowner(
        args.data_dir, args.n_owners, args.valid_size, rng, tgt, args.data_size, args.unbalance)

    val_loader = torch.utils.data.DataLoader(
        dataset=MyDataset(valid_data[0].unsqueeze(1), valid_data[1]) if valid_data[0].dim() == 3 else MyDataset(
            valid_data[0], valid_data[1]),
        shuffle=True,
        batch_size=len(valid_data[0]),
        # pin_memory=True,
    )
    for i in range(args.n_owners):
        scale = i * 1 / args.n_owners
        data_owner_X[i] = addGaussianNoise(data_owner_X[i], scale=scale)


    kwargs = {'share': args.share, 'public_size': args.public_size, 'batch_size': args.batch_size}
    sample_loaders, tgt_index_range = sample_from_dataowner(
        data_owner_X, data_owner_y, args.data_dir, args.n_owners, rng, src, kwargs)

    src_loader, tgt_loader = sample_loaders

    ###################################
    ### only trained on public dataset#
    ###################################
    # logger.info('\n only trained on public dataset')
    # pub_model = get_model(args.model, num_cls=args.num_cls, weights_init=None).to(args.device)
    # optimizer = torch.optim.Adam(pub_model.parameters(), lr=1e-2)
    # best_metrics1 = None
    # for epoch in range(5):
    #     metrics1 = {'Epoch': epoch}
    #     train_metric = train_epoch(src_loader, pub_model, optimizer, flag='[Public] ')
    #     test_metric = test_epoch(val_loader, pub_model)
    #     metrics1.update(train_metric)
    #     metrics1.update(test_metric)
    #     if best_metrics1 is None or (
    #             float(metrics1['Test/Loss']) < float(best_metrics1['Test/Loss']) and
    #             float(best_metrics1['Test/Acc']) < float(metrics1['Test/Acc']) <= 100.0
    #     ):
    #         best_metrics1 = metrics1
    #     # logger.info(metrics1)
    # # print('[Public Dataset]')
    # logger.info(best_metrics1)

    #######################################################
    ### only trained on pre-shared dataset (no public data)
    #######################################################
    # logger.info('\n only trained on pre-shared dataset (no public data)')
    # share_model = get_model(args.model, num_cls=args.num_cls, weights_init=None).to(args.device)
    # optimizer = torch.optim.Adam(share_model.parameters(), lr=1e-3)
    # best_metrics2 = None
    # for epoch in range(args.epochs):
    #     metrics2 = {'Epoch': epoch}
    #     train_metric = train_epoch(tgt_loader, share_model, optimizer, flag='[PreShare] ')
    #     test_metric = test_epoch(val_loader, share_model)
    #     metrics2.update(train_metric)
    #     metrics2.update(test_metric)
    #     if best_metrics2 is None or (
    #             float(metrics2['Test/Loss']) < float(best_metrics2['Test/Loss']) and
    #             float(best_metrics2['Test/Acc']) < float(metrics2['Test/Acc']) <= 100.0
    #     ):
    #         best_metrics2 = metrics2
    #     logger.info(metrics2)
    # logger.info(best_metrics2)

    ####################################################
    ## trained on public dataset and pre-shared dataset#
    ####################################################
    logger.info('\n trained on public dataset and pre-shared dataset')
    # print(args.out_dir)
    args.out_dir="D:\\CODE\\code\\Marketplace\\DataValuation\\outputs\\effective\\models\\usps2minist_share_0_1\\"
    # args.out_dir="../outputs/effective/models/usps2mnist_share_0_1/"
    pretrain_net_file = os.path.join(args.out_dir, '{}_pretrain_{:s}.pth'.format(args.model, src))
    utility_data = collect_from_loader(sample_loaders)
    # print(pretrain_net_file)
    if not os.path.exists(pretrain_net_file):
        pretraining_from_data(utility_data, args.model, args.num_cls, pretrain_net_file, num_epoch=100, batch=32, lr=1e-4)
    else:
        print('Skipping pre-training net training, exists:', pretrain_net_file)

    extractor = get_model('AddaNet',  model=args.model, num_cls=args.num_cls, src_weights_init=pretrain_net_file).to(args.device)
    lr_adda = {'src': args.lr_ext_src, 'tgt': args.lr_ext_tgt, 'dis': args.lr_ext_dis}
    opt_tgt = torch.optim.Adam(extractor.tgt_net.parameters(), lr=lr_adda['tgt'])
    opt_dis = torch.optim.Adam(extractor.discriminator.parameters(), lr=lr_adda['dis'])

    best_metrics3 = None
    for epoch in range(1, 200 + 1):
        metrics3 = {'Epoch': epoch}
        train_gan(src_loader, tgt_loader, extractor, opt_tgt, opt_dis, epoch, logger, verbose=False)

    adda_net_file = os.path.join(args.out_dir, 'adda_{:s}_net_{:s}_{:s}.pth'.format(args.model, src, tgt))
    extractor.save(adda_net_file)

    # print('Evaluating {}->{} adda model: {}'.format(src, tgt, adda_net_file))
    metrc_test_tgt = test_epoch(val_loader, extractor.tgt_net)
    # metrics3.update(metric_test_src)
    metrics3.update(metrc_test_tgt)
    if best_metrics3 is None or (
            float(metrics3['Test/Loss']) < float(best_metrics3['Test/Loss']) and
            float(best_metrics3['Test/Acc']) < float(metrics3['Test/Acc']) <= 100.0
    ):
        best_metrics3 = metrics3
    logger.info(best_metrics3)

    ########################################################
    ### trained on public dataset and weighted full dataset#
    ########################################################
    logger.info('\n trained on public dataset and full dataset')
    pretrain_net_file = os.path.join(args.out_dir, '{}_pretrain_{:s}.pth'.format(args.model, src))
    utility_data = collect_from_loader(sample_loaders)

    if not os.path.exists(pretrain_net_file):
        pretraining_from_data(utility_data, args.model, args.num_cls, pretrain_net_file,
                              num_epoch=40, batch=32, lr=1e-4)
    else:
        print('Skipping pre-training net training, exists:', pretrain_net_file)

    full_X = torch.cat(data_owner_X)
    full_y = torch.cat(data_owner_y)
    tgt_full_loader = torch.utils.data.DataLoader(
        dataset=MyDataset(full_X.unsqueeze(1), full_y) if valid_data[0].dim() == 3 else MyDataset(full_X, full_y),
        shuffle=True,
        batch_size=args.batch_size,
        # pin_memory=True,
    )

    extractor = get_model('AddaNet', model=args.model, num_cls=args.num_cls, src_weights_init=pretrain_net_file).to(
        args.device)
    lr_adda = {'src': args.lr_ext_src, 'tgt': args.lr_ext_tgt, 'dis': args.lr_ext_dis}
    opt_tgt = torch.optim.Adam(extractor.tgt_net.parameters(), lr=lr_adda['tgt'])
    opt_dis = torch.optim.Adam(extractor.discriminator.parameters(), lr=lr_adda['dis'])

    best_metrics4 = None
    for epoch in range(1, 200 + 1):
        metrics4 = {'Epoch': epoch}
        train_gan(src_loader, tgt_loader, extractor, opt_tgt, opt_dis, epoch, logger, verbose=True)

        metrc_test_tgt = test_epoch(val_loader, extractor.tgt_net)
        metrics4.update(metrc_test_tgt)
        if best_metrics4 is None or (
                float(metrics4['Test/Loss']) < float(best_metrics4['Test/Loss']) and
                float(best_metrics4['Test/Acc']) < float(metrics4['Test/Acc']) <= 100.0
        ):
            best_metrics4 = metrics4
        # logger.info(metrics4)


    logger.info(best_metrics4)


    return 0, best_metrics3['Test/Acc']


if __name__ == '__main__':
    args = experimental_setting()
    src = 'usps'
    assert src is not None, "src data can not be None"
    tgt = args.tgt
    args.model = 'LeNet' if tgt == 'mnist' else 'DTN'
    args.num_cls = 10
    proxy_model = logistic_data_to_acc

    args.out_dir = os.path.join(args.out_dir, 'effective/models/{}2{}_share:{}/'.format(src, tgt, args.share))
    logfile = os.path.join(args.out_dir, 'margContrib_{}2{}_{}o{}s.log'.format(src, tgt, args.n_owners, args.data_size))
    if os.path.exists(logfile):
        os.remove(logfile)

    # progbar = tqdm(range(args.repeats), file=sys.stdout, position=0, leave=True)
    with open('margin_contribution.csv', 'a') as f:
        writer_object = writer(f)
        for i in range(args.repeats):
            print('\n')
            args.seed = i
            args.utility_seed = i + 1
            acc1, acc2 = main()

            results = [args.share, acc1, acc2]
            writer_object.writerow(results)
        f.close()

