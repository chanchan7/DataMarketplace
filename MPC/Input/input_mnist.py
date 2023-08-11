"""
More formal setting of data valuation
"""

import numpy as np

import logging
import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from sklearn.model_selection import train_test_split
import copy

import torch
from dataloader import get_data, get_loaders_name, partition_data
from shapley import comb_shapley_uf, comb_shapley_exactly, utility_score
from models.models import get_model
from models.scenario1 import DeepSet
from utils import addGaussianNoise
from sample_utility import sample_utility, sample_utility_unbalance_owner
from utility_models import logistic_data_to_acc

from train_uf import train_uf
from train_src_net import train as train_source

from torchvision import datasets,transforms

logger = logging.getLogger(__name__)

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./datasets', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=1e-5, type=float, help='Random seed')
    parser.add_argument('--betas', default=(0.9, 0.999), help='Random seed')
    parser.add_argument('--weight_decay', default=0, help='Random seed')

    parser.add_argument('--out-dir', default='scenario1-output/square_reduce', type=str, help='Output directory')
    parser.add_argument('--out-feature_dir', default='scenario1-output/feature', type=str, help='feature directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--utility-seed', default=123, type=int, help='Random seed when generating deep sets')
    parser.add_argument('--tgt-sample-seed', default=456, type=int,
                        help='Random seed when sampling from target dataset')
    parser.add_argument('--valid-size', default=300, type=int, help='size of validation data')
    parser.add_argument('--n-owners', default=3, type=int, help='Random seed')
    return parser.parse_args()


def main(tgt):
    args = get_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    src_weights_init = os.path.join(args.out_dir, '{:s}_{:s}_mean_square_reduce.pth'.format('LeNet', tgt))
    print(src_weights_init)
    ds_file = os.path.join(args.out_dir, 'Uf_{:s}_mean_square_reduce.pth'.format(tgt))
    ext_file = os.path.join(args.out_dir, 'LeNet_{:s}_mean_square_reduce.pth'.format(tgt))
    extractor = get_model('LeNet', num_cls=10, weights_init=src_weights_init).cuda()
    extractor.load(ext_file)
    print(extractor)
    hidden_num = 128 if extractor.output_dim == 500 else 512
    deepset = DeepSet(extractor.output_dim, set_features=hidden_num, hidden_ext=hidden_num,
                          hidden_reg=hidden_num).cuda()
    deepset.load_state_dict(torch.load(ds_file))
    print(deepset)
    dic_lenet = torch.load(ext_file)

    print(dic_lenet )
    dic = torch.load(ds_file)
    j = 0
    for i in dic_lenet.keys():
        print(i,dic_lenet[i].size())
    print('---------------------------------')
    for i in dic.keys():
        print(i,dic[i].size())

    w = torch.cat((dic_lenet['conv_params.0.weight'].cpu().view(-1)/255,
                  dic_lenet['conv_params.0.bias'].cpu().view(-1),
                  dic['0.0.weight'].cpu().view(-1),
                  dic['1.0.weight'].cpu().view(-1),
                  dic['linear.weight'].cpu().view(-1),
                  dic['linear.bias'].cpu().view(-1)))


    extractor.eval()
    deepset.eval()
    np.savetxt("w",w,delimiter = "\n")

    transform = transforms.Compose([transforms.ToTensor()
                                ])

    data_train = datasets.MNIST(root = "./datasets/",
                                transform=transform,
                                train = True,
                                download = True)

    data_test = datasets.MNIST(root="./datasets/",
                            transform = transform,
                            train = False)



    # data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
    #                                                 batch_size = 64,
    #                                                 shuffle = True)

    # data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
    #                                                batch_size = 64,
    #                                                shuffle = True)


    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                            batch_size = 1,
                                            shuffle = False)
    # extractor.weight. = dic_lenet['conv_params.0.weight']
    w = torch.Tensor().cuda()
    index = 0 
    for i,j in data_loader_test:
        # print(i)
        index += 1
        score = extractor(i.cuda(),with_ft=1)[1]
        w= torch.cat([w,score.view(1,-1)],0)
        if index>=20:
            break
    print(w.size())
    
    print(score.size())
    print("------------------------------------------")
    score_without = deepset(w.view(1,-1,1,3136))
    print(score_without[0][0])
    print(dic['linear.bias'].cpu().view(-1))
    

if __name__ == '__main__':
    main('mnist')