"""
Get data
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.datasets import load_svmlight_file
# from preprocess import addGaussianNoise
import cv2

def addGaussianNoise(x, scale=1, seed=1):
    torch.random.manual_seed(seed)
    a_norm = x.norm(2, dim=[-2, -1], keepdim=True)
    noise_x = x + torch.normal(0, scale, size=x.shape)
    norm_x = noise_x/noise_x.norm(2, dim=[-2, -1], keepdim=True) * a_norm
    # no = torch.clip(torch.normal(0, scale, size=x.shape), -clip, clip)
    return torch.clip(norm_x, 0, 1)


def get_data(data, dir_):
    if data == 'usps':
        train_dataset = datasets.USPS(dir_, train=True, download=True)
        test_dataset = datasets.USPS(dir_, train=False, download=True)
        x_transfer = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(28),
                                         transforms.ToTensor(),
                                         ])

        x_train = train_dataset.data
        x_test = test_dataset.data
        y_train = torch.tensor(train_dataset.targets, requires_grad=False)
        y_test = torch.tensor(test_dataset.targets, requires_grad=False)

        x_train_r = []
        for i in range(x_train.shape[0]):
            x_r = x_transfer(x_train[i])
            x_train_r.append(x_r)

        x_train_r = torch.cat(x_train_r)

        x_test_r = []
        for i in range(x_test.shape[0]):
            x_r = x_transfer(x_test[i])
            x_test_r.append(x_r)

        x_test_r = torch.cat(x_test_r)

    elif data == 'mnist':
        train_dataset = datasets.MNIST(dir_, train=True, download=True)
        test_dataset = datasets.MNIST(dir_, train=False, download=True)
        # x_transfer = transforms.Compose([transforms.ToPILImage(),
        #                                  transforms.Resize(28),
        #                                  transforms.ToTensor(),
        #                                  transforms.Normalize(0.5, 0.5)
        #                                  ])

        x_train = train_dataset.data
        x_test = test_dataset.data
        y_train = train_dataset.targets
        y_test = test_dataset.targets

        x_train_r = x_train / 255.
        x_test_r = x_test / 255.

    elif data == 'a9a':
        data_path = os.path.join(dir_, 'a9a.t')
        x_train, y_train, _ = load_svmlight_file(data_path, query_id=True)
        x_train_r = torch.tensor(x_train.todense(), dtype=torch.float32)
        y_train = (y_train >= 0).astype(np.float32)
        # y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=False)
        x_test_r = None
        y_test = None
    elif data == 'cifar10':

        train_transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            # transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(dir_, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(dir_, train=False, download=True, transform=train_transform)

        train_loader = DataLoader(train_dataset, batch_size=10000)
        test_loader = DataLoader(test_dataset, batch_size=10000)

        x_train_r, y_train = [], []
        x_test_r, y_test = [], []

        for (image, label) in iter(train_loader):
            x_train_r.append(image)
            y_train.append(label)

        x_train_r = torch.cat(x_train_r)
        y_train = torch.cat(y_train)
        for (image, label) in iter(test_loader):
            x_test_r.append(image)
            y_test.append(label)
        x_test_r = torch.cat(x_test_r)
        y_test = torch.cat(y_test)

    elif data == 'stl10':
        train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.STL10(dir_, 'train', download=True)
        test_dataset = datasets.STL10(dir_, 'test', download=True)

        x_train = train_dataset.data
        x_test = test_dataset.data
        y_train = train_dataset.labels
        y_test = test_dataset.labels
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        x_train_r = []
        for i in range(x_train.shape[0]):
            img = x_train[i]
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            x_r = train_transform(img)
            x_train_r.append(x_r)

        b = torch.Tensor(x_train.shape[0], 3, 32, 32)
        x_train_r = torch.cat(x_train_r, out=b)
        x_train_r = x_train_r.reshape(x_train.shape[0], 3, 32, 32)

        x_test_r = []
        for i in range(x_test.shape[0]):
            x_r = train_transform(x_test[i])
            x_test_r.append(x_r)

        b = torch.Tensor(x_test.shape[0], 3, 32, 32)
        x_test_r = torch.cat(x_test_r, out=b)
        x_test_r = x_test_r.reshape(x_test.shape[0], 3, 32, 32)

    else:
        raise ValueError('Unknown data name')

    return ((x_train_r, y_train),
            (x_test_r, y_test))


def get_loaders_name(name, dir_, batch_size):
    transform = get_transform()
    train_dataset, test_dataset = [], []
    if name == 'usps':
        train_dataset = datasets.USPS(
            dir_, train=True, transform=transform, download=True)
        test_dataset = datasets.USPS(
            dir_, train=False, transform=transform, download=True)
    elif name == 'mnist':
        train_dataset = datasets.MNIST(
            dir_, train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(
            dir_, train=False, transform=transform, download=True)
    elif name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(dir_, train=False, transform=train_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, test_loader


def get_loaders_data(data, batch_size, transform, input_type='cnn'):
    X, y = data
    if input_type == 'cnn' and transform is None:
        if X.dim() == 3:
            X = X.unsqueeze(1)
    elif input_type == 'logistic' and transform is None:
        X = X.view(X.shape[0], -1)
    train_loader = torch.utils.data.DataLoader(
        dataset=MyDataset(X, y, transform),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    return train_loader


def get_data_dataowner(data, dir_, n_owner):
    ((train_X, train_y), (test_X, test_y)) = get_data(data, dir_)
    train_X = torch.tensor(train_X, requires_grad=False)
    train_y = torch.tensor(train_y, requires_grad=False)
    test_X = torch.tensor(test_X, requires_grad=False)
    test_y = torch.tensor(test_y, requires_grad=False)

    rng = np.random.default_rng(303)
    data_owner_X, data_owner_y = partition_data(train_X, train_y, n_owner, rng)
    return (data_owner_X, data_owner_y), (test_X, test_y)


class MyDataset(Dataset):  # 继承Dataset
    def __init__(self, X, y, transform=None):  # __init__是初始化该类的一些基础参数
        self.X = X
        self.y = y.long()

        self.transform = transform

    def __len__(self):  # 返回整个数据集的大小
        return len(self.X)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]

        train_X = self.X[index]
        train_y = self.y[index]
        if self.transform:
            img = Image.fromarray(train_X.numpy(), mode='L')
            train_X = self.transform(img)  # 对样本进行变换
        return train_X, train_y


class MyDataset3(Dataset):  # 继承Dataset
    def __init__(self, X, y, ind, transform=None):  # __init__是初始化该类的一些基础参数
        self.X = X
        self.y = y.long()
        self.ind = ind

        self.transform = transform

    def __len__(self):  # 返回整个数据集的大小
        return len(self.X)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]

        train_X = self.X[index]
        train_y = self.y[index]
        owner_ind = self.ind[index]

        if self.transform:
            img = Image.fromarray(train_X.numpy(), mode='L')
            train_X = self.transform(img)  # 对样本进行变换
        return train_X, train_y, owner_ind


def get_transform():
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(28),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x/255.)
    ])
    return transform


def partition_data(X, y, n_parts, rng, unbalance=False, data_size=None):
    """
    :param unbalance:
    :param data_size: the number of data points each owner holds
    :param X: has the type of tensor
    :param y: has the type of tensor or array
    :param n_parts: number of data owners
    :param rng: random generator
    """
    data_owner_X = [[] for i in range(n_parts)]
    data_owner_y = [[] for i in range(n_parts)]
    num_classes = int(y.max().item()) + 1
    for i in range(num_classes):
        i_samples = torch.where(y == i)[0]
        i_samples_num = len(i_samples)
        if unbalance:
            toss = rng.uniform(0, 1)

            # With probability ub_prob, sample a class-imbalanced subset
            if toss > 1 - 0.8:
                alpha = np.random.choice(range(1, 100), size=n_parts, replace=True)
            else:
                alpha = np.random.choice(range(80, 100), size=n_parts, replace=True)
            p = rng.dirichlet(alpha=alpha)
            data_for_each_owner = np.random.multinomial(i_samples_num, p, 1)[0]
            # data_for_each_owner = np.round(data_for_each_owner).astype(np.int)
            consum = np.cumsum(data_for_each_owner)
            consum = np.hstack([np.array([0]), consum])
            for j in range(n_parts):
                chose = i_samples[consum[j]: consum[j + 1]]
                data_owner_X[j].append(X[chose])
                data_owner_y[j].append(y[chose])

            # data_for_each_owner = data_size * prob_hold
            # data_for_each_owner = np.floor(data_for_each_owner).astype(np.int)
            # consum = np.cumsum(data_for_each_owner)
            # consum = np.hstack([np.array([0]), consum])
            # for j in range(n_parts):
            #     data_owner_X[j].append(X[i_samples[consum[j]: consum[j + 1]]])
            #     data_owner_y[j].append(y[i_samples[consum[j]: consum[j + 1]]])
        else:
            alpha = np.ones(n_parts) * 30
            p = rng.dirichlet(alpha=alpha)
            data_for_each_owner = np.random.multinomial(i_samples_num, p, 1)[0]
            # data_for_each_owner = np.round(data_for_each_owner).astype(np.int)
            consum = np.cumsum(data_for_each_owner)
            consum = np.hstack([np.array([0]), consum])

            size_each_class = data_size // num_classes + 100
            for j in range(n_parts):
                chose = i_samples[consum[j]: consum[j + 1]][:size_each_class]
                data_owner_X[j].append(X[chose])
                data_owner_y[j].append(y[chose])

    for i in range(n_parts):
        data_owner_X[i] = torch.vstack(data_owner_X[i])
        data_owner_y[i] = torch.hstack(data_owner_y[i])
        if unbalance:  # vary the number of data points per data owner
            data_size_ = data_size + rng.integers(-100, 100)
        else:
            data_size_ = data_size  # + rng.integers(-100, 100)
        if data_owner_X[i].shape[0] >= data_size_:
            rand = rng.choice(data_owner_X[i].shape[0], data_size_, replace=False)
            data_owner_X[i] = data_owner_X[i][rand]
            data_owner_y[i] = data_owner_y[i][rand]

    return data_owner_X, data_owner_y


def choose_balance_valid_data(tgt, y, size, rng):
    index = []
    if tgt == 'a9a':
        num_class = 2
    else:
        num_class = 10
    size_for_each_class = size // num_class
    for i in range(num_class):
        i_samples = np.where(y == i)[0]
        index_i = rng.choice(i_samples, size=size_for_each_class, replace=False)
        index.extend(index_i)
    rng.shuffle(index)
    return index


def construct_dataowner(data_dir, n_owners, valid_size, rng, tgt, data_size, unbalance=False):
    ((tgt_train_X, tgt_train_y), (tgt_test_X, tgt_test_y)) = get_data(tgt, data_dir)
    valid_sample_idx = choose_balance_valid_data(tgt, tgt_test_y, valid_size, rng)
    # rng.choice(range(len(tgt_train_X)), size=valid_size, replace=False)
    if not isinstance(tgt_train_X, torch.Tensor):
        tgt_train_X = torch.tensor(tgt_train_X, requires_grad=False)
        tgt_test_X = torch.tensor(tgt_test_X, requires_grad=False)
    if not isinstance(tgt_train_y, torch.Tensor):
        tgt_train_y = torch.tensor(tgt_train_y, requires_grad=False, dtype=torch.long)
        tgt_test_y = torch.tensor(tgt_test_y, requires_grad=False, dtype=torch.long)

    valid_sample_X, valid_sample_y = tgt_test_X[valid_sample_idx], tgt_test_y[valid_sample_idx]
    # owner_sample_idx = np.arange((len(tgt_train_X)))
    # owner_sample_X, owner_sample_y = tgt_train_X[owner_sample_idx], tgt_train_y[owner_sample_idx]

    data_owner_X, data_owner_y = partition_data(tgt_train_X, tgt_train_y, n_owners, rng, unbalance, data_size)
    valid_data = (valid_sample_X, valid_sample_y)

    return data_owner_X, data_owner_y, valid_data


def sample_from_dataowner(data_owner_X, data_owner_y, data_dir, n_owners, rng, src=None, kwargs={}):
    tgt_sample_X, tgt_sample_y = [], []  # To train deepset
    index = []
    index_range = []
    share = kwargs['share']
    for i in range(n_owners):
        if int(share * data_owner_y[i].shape[0]) < 10:
            sample_idx = rng.choice(range(len(data_owner_y[i])), size=int(share * data_owner_y[i].shape[0]),
                                    replace=False)
        elif share == 1.:
            sample_idx = np.arange(len(data_owner_y[i]))
        else:
            sample_idx = choose_balance_valid_data('mnist', data_owner_y[i], int(share * data_owner_y[i].shape[0]), rng)
        index_range.append(len(sample_idx))
        tgt_sample_X.append(data_owner_X[i][sample_idx])
        tgt_sample_y.append(data_owner_y[i][sample_idx])
        index.append(np.ones_like(sample_idx) * i)
    tgt_sample_X = torch.vstack(tgt_sample_X)

    tgt_sample_y = torch.tensor(np.hstack(tgt_sample_y))
    # index = torch.tensor(np.hstack(index))
    # tgt_data = (tgt_sample_X, tgt_sample_y)

    public_size = kwargs['public_size']
    if tgt_sample_X.dim() != 4:
        tgt_sample_dataset = MyDataset(tgt_sample_X.unsqueeze(1), tgt_sample_y, None)
    else:
        tgt_sample_dataset = MyDataset(tgt_sample_X, tgt_sample_y, None)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    tgt_sample_loader = torch.utils.data.DataLoader(
        dataset=tgt_sample_dataset,
        shuffle=True,
        batch_size=min(kwargs['batch_size'], tgt_sample_X.shape[0])
    )

    if src is not None:
        ((src_train_X, src_train_y), (_, _)) = get_data(src, data_dir)
        if not isinstance(src_train_y, torch.Tensor):
            src_train_y = torch.tensor(src_train_y, requires_grad=False)

        # src_train_X[:3500] = addGaussianNoise(src_train_X[:3500], scale=1)

        # index = choose_balance_valid_data(src_train_X, src_train_y, public_size, rng)
        index = rng.choice(np.arange(src_train_X.shape[0]), public_size)
        src_data_sample = (src_train_X[index], src_train_y[index])

        if src_data_sample[0].dim() != 4:
            src_sample_dataset = MyDataset(src_data_sample[0].unsqueeze(1), src_data_sample[1], None)
        else:
            src_sample_dataset = MyDataset(src_data_sample[0], src_data_sample[1], None)
        src_sample_loader = torch.utils.data.DataLoader(
            dataset=src_sample_dataset,
            shuffle=True,
            batch_size=min(kwargs['batch_size'], tgt_sample_X.shape[0]),
        )
    else:
        src_sample_loader = None

    return (src_sample_loader, tgt_sample_loader), index_range


if __name__ == '__main__':
    # ((x_train_r, y_train), (x_test_r, y_test)) = get_data('cifar10', './datasets')
    print()
