import copy

import torch
from utils import featureExtract_cycda
from models.models import get_model
from models.deepsets_net import DeepSets
from deepsets_training import utility_deepset
import torch.optim as optim
from torch import nn
import os
import numpy as np
from utils import metric


def train(x_train_few, train_set, test_set, ds_model, net, opt_src, epoch, logger):
    net.eval()
    X_feature, y_feature = train_set

    # --------------------------------Optimize Deepset network--------------------------------
    # logger.info("[Train Adda Deepset] Epoch: {} Fit deepset proxy_model...".format(epoch))

    x_train_few_ResnetFeature = featureExtract_cycda(x_train_few, net).detach()
    train_loss = ds_model.fit(x_train_few_ResnetFeature, train_set, n_epoch=epoch, batch_size=32, logger=logger)
    test_loss = ds_model.evaluate(x_train_few_ResnetFeature, test_set, batch_size=32, logger=logger)

    # --------------------------------Optimize src feature network--------------------------------
    # net.train()
    # loss_function = nn.MSELoss(reduction='sum')
    # N, K = x_train_few_ResnetFeature.shape
    # ds_loss = 0
    # for i in range(len(y_feature)):
    #     opt_src.zero_grad()
    #     assert len(X_feature[i]) > 0
    #     selected_train_data = featureExtract_cycda(x_train_few[X_feature[i]], net)
    #     selected_train_data = selected_train_data.view(1, -1, K)
    #     y_pred = ds_model.model(selected_train_data)
    #
    #     loss = loss_function(y_pred, y_feature[i].view(1, 1).cuda())
    #     loss.backward()
    #     opt_src.step()
    #     ds_loss += loss.item()
    # ds_loss /= len(y_feature)
    # info_str = "Train Adda Deepset finally Deepset: {:.8f}".format(ds_loss)
    # logger.info(info_str)

    return train_loss, test_loss


def train_uf(tgt, utility_data, train_set, test_set, epochs, pretrain_net_file, out_dir,
             logger, model, ext_file, ds_file, lr_extractor, lr_deepsets, device):
    extractor = get_model(model, num_cls=10, weights_init=pretrain_net_file).to(device)
    extractor.train()
    deepset = get_model('DeepSets', in_features=extractor.output_dim, hidden_ext=512, hidden_reg=512).to(device)
    ds_model = utility_deepset(model=deepset, lr=lr_deepsets, device=device)

    os.makedirs(out_dir, exist_ok=True)
    logger.info('Training Utility Fuction for {}'.format(tgt))

    best_train = np.inf
    best_test = np.inf
    opt = optim.Adam(extractor.parameters(), lr=lr_extractor)
    utility_data = utility_data.to(device)
    for epoch in range(epochs):
        if utility_data.dim() != 4:
            utility_data = utility_data.unsqueeze(1)
        train_loss, test_loss = train(utility_data, train_set, test_set, ds_model, extractor, opt, 2, logger)
        logger.info('Epoch: %s, Train Loss: %s, Test Loss: %s' % (epoch + 1, train_loss, test_loss))

        if train_loss <= best_train and test_loss <= best_test:
            best_epoch = epoch
            best_train = train_loss
            best_test = test_loss
            best_extractor = copy.copy(extractor)
            best_ds_model = copy.copy(ds_model)

    # ----------------------------------------Save proxy_model-------------------------------------------
    logger.info(
        'Epoch:{}, Extractor proxy_model saving to: {}, Deepsets proxy_model saving to:{}'.format(best_epoch, ext_file,
                                                                                                  ds_file))

    torch.save(extractor.state_dict(), ext_file)
    torch.save(ds_model.model.state_dict(), ds_file)

    return best_extractor, best_ds_model.model


def train_accuracy_line(shapley_value_uf, random_uf, data_owner_X, data_owner_y, proxy_model, valid_data, logger):
    ################################## Remove data to observe the performance drop ########################
    n_owners = len(data_owner_X)
    from_samll2big = np.argsort(shapley_value_uf)
    from_big2small = np.argsort(shapley_value_uf)[::-1]
    random_samll2big = np.argsort(random_uf)
    random_big2small = np.argsort(random_uf)[::-1]

    num_data_list = [data_owner_X[i].shape[0] for i in range(len(data_owner_X))]
    # num_data = np.cumsum(num_data_list)
    acu_l2h = np.zeros(15)
    acu_h2l = np.zeros(15)
    acu_random_l2h = np.zeros(15)
    acu_random_h2l = np.zeros(15)

    # for owner in range(n_owners):
    #     data_owner_y[owner] = torch.tensor(data_owner_y[owner])
    for index, i in enumerate(np.arange(0, n_owners, 1)):
        acu_l2h[index] = proxy_model(torch.vstack([data_owner_X[j] for j in from_samll2big[i:]]),
                                     torch.hstack([data_owner_y[j] for j in from_samll2big[i:]]),
                                     valid_data)
        acu_random_l2h[index] = proxy_model(torch.vstack([data_owner_X[j] for j in random_samll2big[i:]]),
                                            torch.hstack([data_owner_y[j] for j in random_samll2big[i:]]),
                                            valid_data)
        acu_h2l[index] = proxy_model(torch.vstack([data_owner_X[j] for j in from_big2small[i:]]),
                                     torch.hstack([data_owner_y[j] for j in from_big2small[i:]]),
                                     valid_data)
        acu_random_h2l[index] = proxy_model(torch.vstack([data_owner_X[j] for j in random_big2small[i:]]),
                                            torch.hstack([data_owner_y[j] for j in random_big2small[i:]]),
                                            valid_data)

    logger.info(acu_l2h.tolist())
    logger.info(acu_random_l2h.tolist())
    logger.info(acu_h2l.tolist())
    logger.info(acu_random_h2l.tolist())

    percentage = num_data_list / np.sum(num_data_list)
    score_low2high = metric(acu_l2h, acu_random_l2h, percentage, low2high=True)
    score_high2low = metric(acu_h2l, acu_random_h2l, percentage, low2high=False)

    return (acu_l2h, acu_random_l2h, acu_h2l, acu_random_h2l), ([score_low2high], [score_high2low])
