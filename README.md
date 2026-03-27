# PEARL
Context-Sensitive Abstractions for RL with Parameterized Actions


This repository contains the official implementation of the paper:

**[Context-Sensitive Abstractions for Reinforcement Learning with Parameterized Actions](https://doi.org/10.1609/aaai.v40i29.39635)**
*Rashmeet Kaur Nayyar<sup>1</sup>, Naman Shah<sup>1,2</sup>, Siddharth Srivastava<sup>1</sup>*
<sup>1</sup>Arizona State University, <sup>2</sup>Brown University
**AAAI 2026** (Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 24522-24531)

## Overview

PEARL (Parameterized Extended State/Action Abstractions for RL) addresses sequential decision-making in environments with **parameterized action spaces** -- problems that require both discrete action selection and continuous parameter tuning. The key idea is to enable RL agents to autonomously learn **state and action abstractions** that progressively refine during training, focusing resolution on critical state-action regions. The approach uses abstraction-driven TD(lambda) and demonstrates superior sample efficiency compared to existing methods across continuous-state, parameterized-action domains.


## Project Structure

```
PEARL/
├── main.py                          # Main entry point
├── run_pearl.sh                     # Shell script for running experiments
├── requirements.txt                 # Python dependencies
├── yamls/                           # Experiment configuration files
├── src/
│   ├── agents/
│   │   ├── pearl.py                 # PEARL algorithm
│   │   └── tdlambda.py              # TD(lambda) agent with abstract state-action spaces
│   ├── abstraction/
│   │   ├── abstraction.py           # State and action abstraction manager
│   │   └── flexible_refinement.py   # Clustering-based flexible refinement
│   ├── data_structures/
│   │   ├── cat.py                   # Clustering Abstraction Tree (CAT)
│   │   ├── abstract_state.py        # Abstract state representation
│   │   ├── abstract_action.py       # Abstract action representation
│   │   ├── qvalue_table.py          # Q-value table
│   │   ├── e_table.py               # Eligibility trace table
│   │   ├── trace.py                 # Transition and trace structures
│   │   └── buffer.py                # TD-error, Q-value, and trace buffers
│   └── misc/
│       ├── env_builder.py           # Environment factory
│       ├── log.py                   # Experiment logging (TensorBoard)
│       ├── visualize.py             # Abstraction visualization
│       └── utils.py                 # Utility functions
└── environments/
    ├── envs/
    │   ├── office_param_actions.py   # Office domain
    │   ├── goal_param_actions.py     # Goal domain
    │   ├── logistics_param_actions.py# Logistics (multi-city) domain
    │   ├── pinball_param_actions.py  # Pinball domain
    │   ├── pinball/                  # Pinball simulator
    │   └── gym_goal/                 # Goal environment (gym wrapper)
    └── maps/                         # Map configurations for domains
```

## Installation

### Prerequisites

- Python 3.8+

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AAIR-lab/PEARL.git
   cd PEARL
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the Goal environment:
   ```bash
   pip install -e environments/envs/gym_goal
   ```

## Usage

### Running Experiments

#### Using the Shell Script

The provided `run_pearl.sh` script runs experiments across multiple trials in parallel. Edit the script to select the desired domain and partitioning strategy:

```bash
# In run_pearl.sh, uncomment the desired domain:
# domains="office"
# domains="pinball"
# domains="logistics"
# domains="goal"

# Select partitioning strategy:
# partitioning="flexible"
# partitioning="uniform"

sh run_pearl.sh
```

Results are saved to `results/<domain>/<method_dir>/trial_<n>/`.

#### Running Directly

Run a single experiment using a YAML configuration file:

```bash
python3 main.py --yaml yamls/<domain>_<partitioning>_0.yaml --trial <trial_number> --result_dir results --method_dir PEARL_flexible_td_v
```

**Examples for each domain:**

```bash
# Office
python3 main.py --yaml yamls/office_flexible_0.yaml --trial 1 --result_dir results --method_dir PEARL_flexible_td_v

# Goal
python3 main.py --yaml yamls/goal_flexible_0.yaml --trial 1 --result_dir results --method_dir PEARL_flexible_td_v

# Pinball
python3 main.py --yaml yamls/pinball_flexible_0.yaml --trial 1 --result_dir results --method_dir PEARL_flexible_td_v

# Logistics
python3 main.py --yaml yamls/logistics_flexible_0.yaml --trial 1 --result_dir results --method_dir PEARL_flexible_td_v
```

To run with **uniform partitioning** instead:

```bash
python3 main.py --yaml yamls/office_uniform_0.yaml --trial 1 --result_dir results --method_dir PEARL_uniform_td_v
```

#### Running Multiple Trials

To run 20 trials for a domain (as used in the paper):

```bash
for trial in $(seq 1 20); do
    python3 main.py --yaml yamls/office_flexible_0.yaml --trial $trial --result_dir results --method_dir PEARL_flexible_td_v > results/office/PEARL_flexible_td_v/trial_${trial}/${trial}.log 2>&1 &
done
```

### Ablations

The YAML configs support ablation studies via the `beta` parameter, which controls the refinement criterion:

| Setting | `init_beta` | `decay_beta_amount` | `min_beta` | Description |
|---------|-------------|---------------------|------------|-------------|
| TD-error + Value (default) | 1.0 | 0.02 | 0.15 | Blends TD-error and value dispersion |
| TD-error only | 1.0 | 0.0 | 1.0 | Uses only TD-error for refinement |
| Value only | 0.0 | 0.0 | 0.0 | Uses only value dispersion for refinement |

To modify, edit the corresponding YAML file or pass as command-line arguments:

```bash
# TD-error only ablation
python3 main.py --yaml yamls/office_flexible_0.yaml --trial 1 --init_beta 1.0 --decay_beta_amount 0.0 --min_beta 1.0 --result_dir results --method_dir PEARL_flexible_td

# Value only ablation
python3 main.py --yaml yamls/office_flexible_0.yaml --trial 1 --init_beta 0.0 --decay_beta_amount 0.0 --min_beta 0.0 --result_dir results --method_dir PEARL_flexible_v
```

## Domains

| Domain | State Space | Actions | Description |
|--------|-------------|---------|-------------|
| **Office** | Continuous (x, y, has_coffee, has_mail) | 4 discrete actions with continuous parameters | Navigate an office to collect coffee and mail, then reach a target |
| **Goal (Soccer)** | Continuous (player, goalie, ball positions/velocities) | 3 actions (kick-to, shoot-goal) with continuous parameters | Score a goal past a goalie on a soccer pitch |
| **Pinball** | Continuous (position + velocity) | Parameterized flipper control | Physics-based pinball navigation through obstacles |
| **Logistics** | Multi-city continuous space | Navigation + inter-city transport | Deliver a package across three cities with airports |

## Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gamma` | Discount factor | 0.99 |
| `alpha` | Learning rate | 0.05 |
| `epsilon_min` | Minimum exploration rate | 0.05 |
| `lamda` | Eligibility trace decay | 0.1 |
| `k_cap` | Max clusters for state abstraction | Domain-specific |
| `k_cap_actions` | Max clusters for action abstraction | Domain-specific |
| `abs_interval` | Refinement interval (episodes) | 100 |
| `init_state_abs_level` | Initial state abstraction level | 1 |
| `init_action_abs_level` | Initial action abstraction level | Domain-specific |
| `kernel` | SVM kernel for flexible refinement | linear/rbf |
| `max_clusters` | Max clusters per refinement step | Domain-specific |

## Results

Experiment results are saved under `results/<domain>/<method_dir>/trial_<n>/` and include:
- Learning curves and evaluation scores (pickle files)
- YAML config copy for reproducibility
- TensorBoard logs (viewable via `tensorboard --logdir results/`)

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{nayyar2026context,
  title     = {Context-Sensitive Abstractions for Reinforcement Learning with Parameterized Actions},
  author    = {Nayyar, Rashmeet Kaur and Shah, Naman and Srivastava, Siddharth},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {29},
  pages     = {24522--24531},
  year      = {2026},
  doi       = {10.1609/aaai.v40i29.39635}
}
```

## Acknowledgments

This research was conducted at the [AAIR Lab](https://aair-lab.github.io/), Arizona State University.

