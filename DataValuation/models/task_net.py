import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from models.models import register_model, init_weights
import numpy as np
from utils import Square


class TaskNet(nn.Module):

    num_channels = 3
    image_size = 32
    name = 'TaskNet'

    "Basic class which does classification."
    def __init__(self, num_cls=10, weights_init=None):
        super(TaskNet, self).__init__()
        self.num_cls = num_cls
        self.setup_net()
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        if weights_init is not None:
            self.load(weights_init)
        else:
            init_weights(self)

    def forward(self, x, with_ft=False):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x_ = self.fc_params(x)
        score = self.classifier(x_)
        if with_ft:
            return score, x
        else:
            return score

    def setup_net(self):
        """Method to be implemented in each class."""
        pass

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict, strict=False)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)


@register_model('MPCLeNet')
class LeNet(TaskNet):
    """Network used for MNIST or USPS experiments."""
    num_channels = 1
    image_size = 28
    name = 'MPCLeNet'
    output_dim = 16 * 14 * 14  # dim of last feature layer

    def setup_net(self):

        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            Square(),
        )
        ### The following two parts does not require running MPC
        self.fc_params = nn.Sequential(
            nn.Linear(self.output_dim, 512),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, self.num_cls)
        )

@register_model('LeNet')
class LeNet(TaskNet):
    """Network used for MNIST or USPS experiments."""
    num_channels = 1
    image_size = 28
    name = 'LeNet'
    output_dim = 500  # dim of last feature layer

    def setup_net(self):
        self.conv_params = nn.Sequential(
            nn.Conv2d(self.num_channels, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_params = nn.Linear(50 * 4 * 4, 500)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(500, self.num_cls)
        )

@register_model('MPCDTN')
class DTNClassifier(TaskNet):
    """Classifier used for SVHN->MNIST Experiment"""

    num_channels = 3
    image_size = 32
    name = 'MPCDTN'
    output_dim = 32 * 8 * 8  # dim of last feature layer

    def setup_net(self):
        self.conv_params = nn.Sequential(
            nn.Conv2d(self.num_channels, 16, kernel_size=5, stride=2, padding=2),
            Square(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            Square(),
        )

        ### The following two parts does not require running MPC
        self.fc_params = nn.Sequential(
            nn.Linear(self.output_dim, 512),
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(512, self.num_cls)
        )


@register_model('Logistic')
class Logistic(TaskNet):

    # num_channels = 3
    name = 'Logistic'
    input_dim = 123
    output_dim = 16  # dim of last feature layer

    def setup_net(self):
        self.conv_params = nn.Sequential(
            nn.Linear(123, 16),
            # nn.BatchNorm2d(64),
            # nn.Dropout2d(0.1),
            Square(),
        )

        self.classifier = nn.Sequential(
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(16, self.num_cls)
        )

@register_model('LDTN')
class LDTN(nn.Module):
    num_channels = 3
    image_size = 32
    name = 'LDTN'
    output_dim = 32 * 8 * 8 + 10

    "Basic class which does classification."

    def __init__(self, num_cls=10, weights_init=None):
        super(LDTN, self).__init__()
        self.num_cls = num_cls
        self.criterion = nn.CrossEntropyLoss()

        self.conv_params = nn.Sequential(
            nn.Conv2d(self.num_channels, 16, kernel_size=5, stride=2, padding=2),
            Square(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            Square(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8 + 10, self.num_cls)
        )
        if weights_init is not None:
            self.load(weights_init)
        else:
            init_weights(self)

    def forward(self, x, y, with_ft=False):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        y = F.one_hot(y, num_classes=10)
        x = torch.hstack([x, y])
        score = self.classifier(x)  # bs * num_cls
        if with_ft:
            return score, x
        else:
            return score

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)


if __name__ == '__main__':
    a = LeNet()
    print(a.output_dim)
