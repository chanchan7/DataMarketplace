import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
models = {}


def register_model(name):
    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def get_model(name, **args):
    net = models[name](**args)
    if torch.cuda.is_available():
        net = net.cuda()
    return net


def init_weights(obj):
    for m in obj.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()


class model_for_mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.linear1 = nn.Linear(32 * 7 * 7, 500)
        self.linear2 = nn.Linear(500, 10)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        init_weights(self)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class model_for_cifar10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.linear1 = nn.Linear(64 * 4 * 4, 500)
        self.linear2 = nn.Linear(500, 10)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

        init_weights(self)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Classifier(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(Classifier, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=False),
            nn.ReLU(),
        )
        self.linear = nn.Linear(hidden_features, 10)
        self.criterion = nn.MSELoss(reduction='sum')

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.linear(x)
        return x

class Integrater(nn.Module):
    def __init__(self, extractor: nn.Module, classifier: nn.Module, with_ft=True):
        super(Integrater, self).__init__()
        self.extractor = extractor
        self.classifier = classifier
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x):
        score, x = self.extractor(x, with_ft=True)
        x = self.classifier(x)
        return x