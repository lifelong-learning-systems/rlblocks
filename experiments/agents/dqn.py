import tella
from torch import nn


def dqn(model: nn.Module) -> tella.ContinualRLAgent:
    ...


def dqn_with_memory(model: nn.Module):
    ...


def dqn_with_ewc(model: nn.Module):
    ...


def dqn_with_memory_and_ewc(model: nn.Module):
    ...
