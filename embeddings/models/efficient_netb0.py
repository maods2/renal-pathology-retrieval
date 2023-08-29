from efficientnet_pytorch import EfficientNet
from torch import nn
import torch

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def get_efficient_netb0():
    model = EfficientNet.from_pretrained('efficientnet-b0')
    # model._avg_pooling = Identity()
    # model._dropout = Identity()
    model._fc = Identity()
    # model._swish = Identity()
    model.eval()
    return model


