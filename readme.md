rlblocks
===========

rlblocks provides building blocks for reinforcement learning (RL) agents.


### Example RL Algorithms

1. PPO (Proximal Policy Optimization) in CartPole-v1: [examples/ppo.py](examples/ppo.py). Run with `python -m examples.ppo`
2. DQN (Deep Q Learning) in CartPole-v1: [examples/dqn.py](examples/dqn.py). Run with `python -m examples.dqn`

### MiniGrid Crossing Experiment

We provide an experiment in the MiniGrid environment called MiniGrid Crossing.
See [minigrid_crossing_experiment](minigrid_crossing_experiment) for more details and plots!

To run the experiment run:

`python -m minigrid_crossing_experiment`

Requirements
----------------
* Python 3.7 or greater

Install
-------------
1. Create a conda or virtual environment and activate it

2. Update pip and wheel in your environment:
  ```
  pip install -U pip wheel
  ```
3. Clone this repository:
   ```
   git clone git@github.com:darpa-l2m/rlblocks.git
   ```
   or
   ```
   git clone https://github.com/darpa-l2m/rlblocks.git
   ```
4. Install the rlblocks package and its dependencies:
   ```
   pip install "./rlblocks"
   ```

To update rlblocks, pull the latest changes from the git repository and upgrade:
```
pip install -U .
```


Bug Reports and Feature Requests
---------------------------------
Bug reports and feature requests should be made through issues on Github.

A bug report should contain:
 * descriptive title
 * environment (python version, operating system if install issue)
 * expected behavior
 * actual behavior
 * stack trace if applicable
 * any input parameters need to reproduce the bug

A feature request should describe what you want to do but cannot
and any recommendations for how this new feature should work.


For Developers
----------------
To install rlblocks in editable mode with our development requirements:
```
pip install -e ".[dev]"
```

To run unit tests:
```
pytest
```
For running in conda environment:
```
python -m pytest 
```

To check for PEP8 compliance:
```
black --check rlblocks
```

To autoformat for PEP8 compliance:
```
black rlblocks
```

License
-------

See [LICENSE](LICENSE) for license information.

Acknowledgments
----------------
rlblocks was created by the Johns Hopkins University Applied Physics Laboratory.

![apl logo](apl_small_logo.png)

This software was funded by the DARPA Lifelong Learning Machines (L2M) Program.

The views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.

© 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC
