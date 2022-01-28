import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from copy import deepcopy
import re
from model_efficientnet import *


class EnsembleModel(nn.Module):
    def __init__(
        self, model_1, model_2, model_3, out_size, device, feature=1024, training=True
    ):
        super().__init__()
        self.model_1 = model_1.to(device=device)
        self.model_2 = model_2.to(device=device)
        self.model_3 = model_3.to(device=device)
        self.fc1 = nn.Linear(1024 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, out_size)

    def forward(self, img):

        feature_1 = self.model_1(img)
        feature_2 = self.model_2(img)
        feature_3 = self.model_3(img)

        feature_ = torch.cat([feature_1, feature_2, feature_3], dim=1)
        out1 = self.fc1(feature_)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2)
        out = F.softmax(out3, dim=-1)

        return out
