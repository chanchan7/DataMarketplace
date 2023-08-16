"""
Generating DeepSets
"""
import numpy as np
import torch
from dataloader import get_data, get_loaders_data
from proxy_models import logistic_data_to_acc, cnn_data_to_acc, Small_models
from utils import sample_count
import os
import copy


def sample_utility(n, size_min, size_max, utility_func, utility_func_args, random_state, verbose=True, kwargs={}):
    x_train, y_train = utility_func_args[0]

    X_feature = []
    y_feature = []

    N = len(y_train)

    np.random.seed(random_state)

    for num in range(n):

        n_select = np.random.choice(range(size_min, size_max))

        subset_index = []

        """
        if unbalance:
          n_per_class = int(N / 10)
          alpha = np.ones(10)
          alpha[np.random.choice(range(10))] = np.random.choice(range(1, 50))
        else:
          alpha = np.random.choice(range(1, 20), size=10, replace=True)
        """

        # toss = np.random.uniform()
        # With probability ub_prob, sample a class-imbalanced subset
        # if toss > 1 - ub_prob:
        #
        #     alpha = np.ones(10)
        #     alpha[np.random.choice(range(10))] = np.random.choice(range(1, 50))
        # else:
        #     alpha = np.random.choice(range(90, 100), size=10, replace=True)
        # alpha = np.ones(10) * 30
        # p = np.random.dirichlet(alpha=alpha)
        p = 1 / 10 * np.ones(10)
        occur = np.random.choice(range(10), size=n_select, replace=True, p=p)
        counts = np.array([np.sum(occur == i) for i in range(10)])

        # p = np.random.uniform(0, 1)
        # if p < 0.33:
        for i in range(10):
            ind_i = np.where(y_train == i)[0]
            if len(ind_i) > counts[i]:
                selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=False)
            else:
                selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=True)
            subset_index = subset_index + list(selected_ind_i)
        # elif 0.33<=p<=0.66:
        #     for i in range(10):
        #         ind_i = np.where(y_train == i)[0]
        #         ind_i = ind_i[ind_i<=3500]
        #         if len(ind_i) > counts[i]:
        #             selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=False)
        #         else:
        #             selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=True)
        #         subset_index = subset_index + list(selected_ind_i)
        # else:
        #     for i in range(10):
        #         ind_i = np.where(y_train == i)[0]
        #         ind_i = ind_i[ind_i>3500]
        #         if len(ind_i) > counts[i]:
        #             selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=False)
        #         else:
        #             selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=True)
        #         subset_index = subset_index + list(selected_ind_i)


        subset_index = np.array(subset_index)
        a = (subset_index<=3500).sum()
        # print(a)
        valid_acc = utility_func(x_train[subset_index], y_train[subset_index], utility_func_args[1], kwargs=kwargs)
        if verbose:
            print('{} / {}, n_select: {}, label: {}'.format(num+1, n, a, valid_acc))
        y_feature.append(valid_acc)

        X_feature.append(subset_index)

    return X_feature, y_feature


def sample_utility_unbalance_owner(n, size_min, size_max, utility_func, utility_func_args, range_index, random_state,
                                   ub_prob=0.8, verbose=False, transfer=False, kwargs={}):
    x_train, y_train = utility_func_args[0]

    sample_max = size_max # min(size_max, x_train.shape[0])
    # sample_min = size_min
    # if sample_max == size_max:
    sample_min = sample_max - 10


    n_owner = len(range_index)
    consum = np.cumsum(range_index)
    consum = np.hstack([np.array([0]), consum])

    X_feature = []
    y_feature = []

    np.random.seed(random_state)

    for num in range(n):

        n_select = np.random.choice(range(sample_min, sample_max))

        subset_index = []

        count_owner = sample_count(n_select, n_owner, ub_prob=ub_prob)
        for j in range(n_owner):
            ind_ij = np.arange(consum[j], consum[j + 1])
            if len(ind_ij) >= count_owner[j]:
                selected_ind_j = np.random.choice(ind_ij, size=count_owner[j], replace=False)
            elif count_owner[j] > len(ind_ij) > 0:
                selected_ind_j = np.random.choice(ind_ij, size=count_owner[j], replace=True)
            else:
                selected_ind_j = []
            subset_index = subset_index + list(selected_ind_j)
        subset_index = np.array(subset_index)

        # subset_index = np.random.choice(y_train.shape[0], n_select)

        if transfer:
            src_model_path = kwargs['src_model_path']
            if not os.path.exists(src_model_path):
                train_loader = get_loaders_data(kwargs['src_data_sample'], 128, None, 'logistic')
                model = Small_models('logistic')
                model.fit(train_loader, 10, 0.01)
                torch.save(model.model.state_dict(), src_model_path)
            valid_acc = utility_func(src_model_path, x_train[subset_index], y_train[subset_index], utility_func_args[1], kwargs)
        else:
            valid_acc = utility_func(x_train[subset_index], y_train[subset_index], utility_func_args[1], kwargs=kwargs)
        if verbose:
            print('{} / {}, percent: {:0.3f}, label: {}'
                    .format(num+1, n, sum(subset_index < range_index[0]) / len(subset_index) * 100, valid_acc))
        y_feature.append(valid_acc)

        X_feature.append(subset_index)

    return X_feature, y_feature


def sample_utility_tf(n, size_min, size_max, utility_func, src_model_path, all_data, random_state, ub_prob=0.2,
                      verbose=False):
    src_x_train, src_y_train = all_data[0]
    tgt_x_train, tgt_y_train = all_data[1]

    if not os.path.exists(src_model_path):
        train_loader = get_loaders_data((src_x_train, src_y_train), 128, None, 'logistic')
        model = Small_models('logistic')
        model.fit(train_loader, 10, 0.01)
        torch.save(model.model.state_dict(), src_model_path)

    X_feature = []
    y_feature = []

    N = len(tgt_y_train)

    np.random.seed(random_state)

    for num in range(n):
        # copy_model = copy.deepcopy(proxy_model)
        n_select = np.random.choice(range(size_min, size_max))

        subset_index = []

        """
        if unbalance:
          n_per_class = int(N / 10)
          alpha = np.ones(10)
          alpha[np.random.choice(range(10))] = np.random.choice(range(1, 50))
        else:
          alpha = np.random.choice(range(1, 20), size=10, replace=True)
        """

        toss = np.random.uniform()

        # With probability ub_prob, sample a class-imbalanced subset
        if toss > 1 - ub_prob:
            n_per_class = int(N / 10)
            alpha = np.ones(10)
            alpha[np.random.choice(range(10))] = np.random.choice(range(1, 50))
        else:
            alpha = np.random.choice(range(90, 100), size=10, replace=True)

        p = np.random.dirichlet(alpha=alpha)
        occur = np.random.choice(range(10), size=n_select, replace=True, p=p)
        counts = np.array([np.sum(occur == i) for i in range(10)])

        for i in range(10):
            ind_i = np.where(tgt_y_train == i)[0]
            if len(ind_i) > counts[i]:
                selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=False)
            else:
                selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=True)
            subset_index = subset_index + list(selected_ind_i)

        subset_index = np.array(subset_index)

        valid_acc = utility_func(src_model_path, tgt_x_train[subset_index], tgt_y_train[subset_index], all_data[2])
        if verbose:
            print(
                '{} / {}, percent: {:0.3f}, label: {}'.format(num, n, sum(subset_index < 199) / len(subset_index) * 100,
                                                              valid_acc))
        y_feature.append(valid_acc)
        X_feature.append(subset_index)

    return X_feature, y_feature


def sample_utility_veryub(n, size_min, size_max, utility_func, utility_func_args, random_state, ub_prob=0.2,
                          verbose=False):
    # very unbalance
    x_train, y_train, x_val, y_val = utility_func_args

    X_feature_test = []
    y_feature_test = []

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    N = len(y_train)

    np.random.seed(random_state)

    for i in range(n):
        if verbose:
            print('{} / {}'.format(i, n))

        n_select = np.random.choice(range(size_min, size_max))

        if n_select > 0:
            subset_index = []

            toss = np.random.uniform()
            # With probability ub_prob, sample a class-imbalanced subset
            if toss > 1 - ub_prob:
                alpha = np.random.choice(range(1, 100), size=10, replace=True)
            else:
                alpha = np.random.choice(range(90, 100), size=10, replace=True)

            p = np.random.dirichlet(alpha=alpha)

            occur = np.random.choice(range(10), size=n_select, replace=True, p=p)
            counts = np.array([np.sum(occur == i) for i in range(10)])

            for i in range(10):
                ind_i = np.where(np.argmax(y_train, 1) == i)[0]
                if len(ind_i) > counts[i]:
                    selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=False)
                else:
                    selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=True)
                subset_index = subset_index + list(selected_ind_i)

            subset_index = np.array(subset_index)

            y_feature_test.append(utility_func(x_train[subset_index], y_train[subset_index], x_val, y_val))
            X_feature_test.append(subset_index)
        else:
            y_feature_test.append(0.1)
            X_feature_test.append(np.array([]))

    return X_feature_test, y_feature_test


def utility_ds(src, n_samples, data_dir, logger, seed):
    ((src_train_X, src_train_y), (src_test_X, src_test_y)) = get_data(src, data_dir)

    # ------------------------------------Generate DeepSets------------------------------------
    x_train_few = torch.zeros((30 * 10, 28, 28))
    y_train_few = torch.zeros((30 * 10), dtype=torch.long)

    idx_few = np.zeros(len(src_train_y))

    # Random Seed: 302, 303, 304
    np.random.seed(303)
    for i in range(10):
        idx = torch.where(src_train_y == i)[0]
        idx_i = np.random.choice(idx, size=30, replace=False)
        x_train_few[i * 30: (i + 1) * 30] = src_train_X[idx_i]
        y_train_few[i * 30: (i + 1) * 30] = src_train_y[idx_i]
        idx_few[i * 30: (i + 1) * 30] = idx_i

    N_val = 300
    np.random.seed(303)
    ind = np.random.choice(range(len(src_test_y)), size=N_val, replace=False)
    x_val_few, y_val_few = src_test_X[ind], src_test_y[ind]
    X_feature, y_feature = sample_utility(n=n_samples, size_min=5, size_max=300,
                                          utility_func=logistic_data_to_acc,
                                          utility_func_args=(x_train_few, y_train_few, x_val_few, y_val_few),
                                          random_state=seed, verbose=True)
    logger.info('DeepSets generated')

    return X_feature, y_feature, x_train_few


def generate_deepsets(train_X, train_y, test_X, test_y, n_samples, logger, seed):
    # ((src_train_X, src_train_y), (src_test_X, src_test_y)) = get_data(src, data_dir)

    # ------------------------------------Generate DeepSets------------------------------------
    x_train_few = torch.zeros((300 * 10, 28, 28))
    y_train_few = torch.zeros((300 * 10), dtype=torch.long)

    rng = np.random.default_rng(seed)

    idx_few = np.zeros(len(train_y))

    # Random Seed: 302, 303, 304
    np.random.seed(303)
    for i in range(10):
        idx = torch.where(train_y == i)[0]
        idx_i = np.random.choice(idx, size=300, replace=False)
        x_train_few[i * 300: (i + 1) * 300] = train_X[idx_i]
        y_train_few[i * 300: (i + 1) * 300] = train_y[idx_i]
        idx_few[i * 300: (i + 1) * 300] = idx_i

    X_feature, y_feature = sample_utility(n=n_samples, size_min=5, size_max=3000,
                                          utility_func=logistic_data_to_acc,
                                          utility_func_args=(x_train_few, y_train_few, test_X, test_y),
                                          random_state=seed, verbose=True)
    logger.info('DeepSets generated')

    return X_feature, y_feature, x_train_few


def sample_data(src_data, data_num, logger, seed):
    # ------------------------------------Generate DeepSets------------------------------------

    src_train_X, src_train_y = src_data

    num_per_class = data_num // 10
    x_train_few = torch.zeros((data_num, 28, 28))
    y_train_few = torch.zeros(data_num, dtype=torch.long)

    idx_few = np.zeros(len(src_train_y))

    # Random Seed: 302, 303, 304
    rng = np.random.default_rng(seed)
    idx_i = rng.choice(np.arange(src_train_y.shape[0]), size=data_num, replace=False)
    x_train_few[:] = src_train_X[idx_i]
    y_train_few[:] = src_train_y[idx_i]

    # for i in range(10):
    #     idx = torch.where(src_train_y == i)[0]
    #     idx_i = rng.choice(idx, size=num_per_class, replace=False)
    #     x_train_few[i * num_per_class: (i + 1) * num_per_class] = src_train_X[idx_i]
    #     y_train_few[i * num_per_class: (i + 1) * num_per_class] = src_train_y[idx_i]
    #     idx_few[i * num_per_class: (i + 1) * num_per_class] = idx_i

    logger.info('DeepSets generated, seed: {}'.format(seed))

    return x_train_few, y_train_few
