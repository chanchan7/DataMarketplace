"""
Utility function (Deespsets) models
"""
import torch
import torch.nn as nn
import torch.optim as optim


class utility_deepset(object):

    def __init__(self, model=None, lr=1e-5, device=None):

        """
        if proxy_model is None:
          self.proxy_model = DeepSet(in_dims).cuda()
        else:
          self.proxy_model = proxy_model.cuda()
        """

        self.model = model
        # self.model.linear.bias = torch.nn.Parameter(torch.tensor([-2.1972]))
        # self.model.linear.bias.requires_grad = False

        self.opt = optim.Adam(self.model.parameters(), lr)
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss(reduction='sum')
        self.device = device

    def fit(self, utility_Feature, train_set, n_epoch, batch_size=32, logger=None):
        # scheduler = StepLR(self.optim, step_size=10, gamma=0.1)
        utility_Feature.requires_grad = False
        # scheduler = MultiStepLR(self.opt, milestones=[10, 15], gamma=0.1)
        X_feature, y_feature = train_set
        train_size = len(y_feature)

        for epoch in range(n_epoch):
            train_loss = 0
            num_batches = train_size // batch_size
            for j in range(num_batches):
                start_ind = j * batch_size
                batch_X, batch_y = [], []
                for i in range(start_ind, min(start_ind + batch_size, train_size)):
                    b = torch.zeros([1000, utility_Feature.shape[1]]).to(utility_Feature.device)
                    selected_train_data = utility_Feature[X_feature[i]]
                    b[:len(X_feature[i])] = selected_train_data

                    batch_X.append(b)
                    batch_y.append([y_feature[i]])

                batch_X, batch_y = torch.stack(batch_X).to(self.device), torch.tensor(batch_y).to(self.device)

                self.opt.zero_grad()
                y_pred = self.model(batch_X)

                loss = self.l2(y_pred, batch_y)
                loss.backward()
                self.opt.step()
                train_loss += loss.item()
            train_loss /= train_size
            # scheduler.step()
            # logger.info('Epoch: %s Train Loss: %s' % (epoch+1, train_loss))
        return train_loss


    def evaluate(self, utility_Feature, valid_set, batch_size, logger):
        utility_Feature.requires_grad = False
        X_feature_test, y_feature_test = valid_set


        test_size = len(y_feature_test)

        test_loss = 0
        num_batches = test_size // batch_size
        for j in range(num_batches):
            start_ind = j * batch_size
            batch_X, batch_y = [], []
            for i in range(start_ind, min(start_ind + batch_size, test_size)):
                b = torch.zeros([1000, utility_Feature.shape[1]])
                assert len(X_feature_test[i]) > 0
                selected_train_data = utility_Feature[X_feature_test[i]]
                b[:len(X_feature_test[i])] = selected_train_data

                batch_X.append(b)
                batch_y.append([y_feature_test[i]])

            batch_X, batch_y = torch.stack(batch_X).to(self.device), torch.tensor(batch_y).to(self.device)

            y_pred = self.model(batch_X)
            loss = self.l2(y_pred, batch_y)
            test_loss += loss.item()
        test_loss /= test_size
        # logger.info('Test Loss: %s' % (test_loss))
        return test_loss

