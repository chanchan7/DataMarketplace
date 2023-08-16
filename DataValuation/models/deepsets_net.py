import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Square
from models.models import register_model, init_weights


@register_model('DeepSets')
class DeepSets(nn.Module):
    def __init__(self, in_features, hidden_ext=128, hidden_reg=128):
        super(DeepSets, self).__init__()
        self.in_features = in_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_ext, bias=False),
            Square(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_reg, 1, bias=False),
            # Square(),
            # nn.Linear(int(hidden_reg/2), 1)
        )

        self.sigmoid = nn.Sigmoid()

        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        init_weights(self)

    def forward(self, x):
        x = self.feature_extractor(x)
        count = (x.clone().detach().requires_grad_(False) != 0).sum(1)
        x = x.sum(1) / count
        x = self.regressor(x)
        x = torch.sigmoid(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'Feature Exctractor=' + str(self.feature_extractor) \
               + '\n Set Feature' + str(self.regressor) + ')'

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)



