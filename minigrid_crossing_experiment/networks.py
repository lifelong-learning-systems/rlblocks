"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
from torch import nn
from torch.nn import functional as tnnf


class MiniGridIndexToOneHot(nn.Module):
    """
    Standard observations in MiniGrid are 7x7x3, describing a 7x7 grid of cells
    in 3 channels: object index [0,10], color index [0,5], and state index [0,2].
    This layer converts those to one-hot encodings. It also permutes the result
    to the expected channels-first format.
    """

    def forward(self, x):
        return (
            torch.cat(
                [
                    tnnf.one_hot(x[..., 0].type(torch.int64), num_classes=11),
                    tnnf.one_hot(x[..., 1].type(torch.int64), num_classes=6),
                    tnnf.one_hot(x[..., 2].type(torch.int64), num_classes=3),
                ],
                dim=-1,
            )
            .permute(0, 3, 1, 2)
            .type(torch.float32)
        )


class MiniGridObjectIndexToOneHot(nn.Module):
    """
    One-hot encodings from index as in :class:MiniGridIndexToOneHot, but here
    using only the object type, not color or state
    """

    def forward(self, x):
        return (
            tnnf.one_hot(x[..., 0].type(torch.int64), num_classes=11)
            .permute(0, 3, 1, 2)
            .type(torch.float32)
        )


class MiniGridCenteredFullObsIndexToOneHot(nn.Module):
    """
    "Full" observations in MiniGrid are NxNx3, describing a NxN grid of cells
    in 3 channels: object index [0,10], color index [0,5], and state index [0,3].
    This layer converts the objects to one-hot encodings. It also permutes the result
    to the expected channels-first format.
    """

    def forward(self, x):
        return (
            tnnf.one_hot(x[..., 0].type(torch.int64), num_classes=11)
            .permute(0, 3, 1, 2)
            .type(torch.float32)
        )


# Network for minigrid
def make_minigrid_net() -> nn.Module:
    return nn.Sequential(
        MiniGridCenteredFullObsIndexToOneHot(),
        nn.Conv2d(11, 5, (1, 1)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(7 * 7 * 5, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 3),
    )
