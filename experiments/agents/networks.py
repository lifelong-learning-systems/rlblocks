import torch
from torch import nn
from torch.nn import functional as tnnf


# MLP network for cartpole
def make_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )


class MiniGridIndexToOneHot(nn.Module):
    """
    Standard observations in MiniGrid are 7x7x3, describing a 7x7 grid of cells
    in 3 channels: object index [0,10], color index [0,5], and state index [0,2].
    This layer converts those to one-hot encodings. It also permutes the result
    to the expected channels-first format.
    """
    def forward(self, x):
        return torch.cat([
            tnnf.one_hot(x[..., 0].type(torch.int64), num_classes=11),
            tnnf.one_hot(x[..., 1].type(torch.int64), num_classes=6),
            tnnf.one_hot(x[..., 2].type(torch.int64), num_classes=3),
        ], dim=-1).permute(0, 3, 1, 2).type(torch.float32)


class MiniGridObjectIndexToOneHot(nn.Module):
    """
    One-hot encodings from index as in :class:MiniGridIndexToOneHot, but here
    using only the object type, not color or state
    """
    def forward(self, x):
        return tnnf.one_hot(x[..., 0].type(torch.int64), num_classes=11).permute(0, 3, 1, 2).type(torch.float32)


# CNN network for minigrid
def make_cnn() -> nn.Module:
    return nn.Sequential(
        MiniGridObjectIndexToOneHot(),
        nn.Conv2d(11, 3, (1, 1)),  # 5x5x11 -> 5x5x3
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(5*5*3, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
        nn.Sigmoid(),  # Assuming Q values in [0, 1]
    )
