import torch
from utils import featureExtract_cycda, featureExtract_cycday
from models.models import get_model
from models.deepsets_net import DeepSets
from deepsets_training import utility_deepset
import torch.optim as optim
from torch import nn
import os
from torch.optim.lr_scheduler import StepLR


def train(x_train_few, y_train_few, train_set, test_set, ds_model, net, opt_src, epoch, ds_criterion, logger):
    net.eval()
    X_feature, y_feature = train_set

    # --------------------------------Optimize Deepset network--------------------------------
    logger.info("[Train Adda Deepset] Epoch: {} Fit deepset proxy_model...".format(epoch))
    x_train_few_ResnetFeature = featureExtract_cycday(x_train_few, y_train_few, net).detach()

    # logger.info(x_train_few_ResnetFeature[0])
    ds_model.fit(x_train_few_ResnetFeature, train_set, n_epoch=10, batch_size=32, logger=logger)
    # ds_model.evaluate(x_train_few_ResnetFeature, test_set, 2, logger)

    # --------------------------------Optimize src feature network--------------------------------
    net.train()

    N, K = x_train_few_ResnetFeature.shape
    ds_loss = 0
    regular_loss = 0
    for i in range(len(y_feature)):
        opt_src.zero_grad()
        assert len(X_feature[i]) > 0
        selected_train_data = featureExtract_cycday(x_train_few[X_feature[i]], y_train_few[X_feature[i]], net)
        selected_train_data = selected_train_data.view(1, -1, K)
        y_pred = ds_model.proxy_model(selected_train_data)

        loss = ds_criterion(y_pred, y_feature[i].view(1, 1).cuda())
        loss.backward()
        opt_src.step()
        ds_loss += loss
    ds_loss /= len(y_feature)
    regular_loss /= len(y_feature)

    info_str = "Train Adda Deepset finally Deepset: {:.8f}".format(ds_loss.item())
    logger.info(info_str)

    return ds_model, net


def train_uf_y(tgt, tgt_sample_X, tgt_sample_y, train_set, test_set, epochs, src_weights,
             out_dir, logger, weight_decay, betas, model, ext_file, ds_file):
    # ext_file = os.path.join(out_dir, 'LeNet_{:s}_mean_relu.pth'.format(tgt))
    # ds_file = os.path.join(out_dir, 'Uf_{:s}_mean_relu.pth'.format(tgt))

    # (X_feature, y_feature), (X_feature_test, y_feature_test) = train_set, test_set
    extractor = get_model(model, num_cls=10, weights_init=src_weights).cuda()
    extractor.train()

    hidden_num = 128 if extractor.output_dim == 500 else 512
    deepset = DeepSets(extractor.output_dim, set_features=hidden_num, hidden_ext=hidden_num, hidden_reg=hidden_num).cuda()
    ds_model = utility_deepset(model=deepset)

    logger.info('Training Utility Fuction for {}'.format(tgt))
    ds_criterion = nn.MSELoss(reduction='sum')

    opt = optim.Adam(extractor.parameters(), lr=1e-7, weight_decay=weight_decay, betas=betas)
    for epoch in range(epochs):
        if tgt_sample_X.dim() != 4:
            tgt_sample_X = tgt_sample_X.unsqueeze(1)
        ds_model, extractor = train(tgt_sample_X, tgt_sample_y, train_set, test_set, ds_model,
                                    extractor, opt, epoch, ds_criterion, logger)

    os.makedirs(out_dir, exist_ok=True)

    logger.info('Extractor proxy_model saving to: {}'.format(ext_file))
    torch.save(extractor.state_dict(), ext_file)

    logger.info("Deep set proxy_model saving to: {}".format(ds_file))
    torch.save(ds_model.model.state_dict(), ds_file)

    return extractor, ds_model.model
