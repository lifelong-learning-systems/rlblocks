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
from rlblocks.base import (
    Observation,
    Action,
    Reward,
    Done,
    Transition,
    TransitionObserver,
    RLTimeUnit,
    Steps,
    Episodes,
    Timer,
    Duration,
    Every,
    RewardTracker,
    ActionTracker,
    PeriodicCallbacks,
    run_env_interaction,
)
from rlblocks.torch_blocks import (
    PolyakParameterUpdate,
    TorchBatch,
    collate,
    ArgmaxAction,
    ChooseBetween,
    Interpolate,
    SoftParameterUpdate,
    PolyakParameterUpdate,
    HardParameterUpdate,
    batch_normalize_advantange,
    NumpyToTorchConverter,
    SampleAction,
)
from rlblocks.loss_functions import (
    TorchBatch,
    QLoss,
    DoubleQLoss,
    PolicyGradientLoss,
    ClippedSurrogatePolicyGradientLoss,
    TDAdvantageLoss,
    ElasticWeightConsolidationLoss,
    SlicedCramerPreservation,
)
from rlblocks.replay import (
    PriorityFn,
    TransitionDatasetWithMaxCapacity,
    TransitionDatasetWithAdvantage,
    BatchGenerator,
    BatchSampler,
    UniformRandomBatchSampler,
    MetaBatchSampler,
)
