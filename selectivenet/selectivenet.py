import torch
import torch.nn.functional as F
import torch.nn as nn


class SelectiveNet(nn.Module):

    def __init__(self, feature: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.feature = feature
        self.f = nn.Linear(feature_dim, num_classes)
        self.g = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        self.h = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.feature(x)
        return (self.f(x), self.g(x), self.h(x), x)
