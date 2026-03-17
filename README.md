# Spiking Temporal Critic (STC)

A research framework for training neuromorphic agents with **Proximal Policy Optimisation (PPO)**, featuring a novel _timing-based_ Spiking Neural Network (SNN) critic that encodes value estimates via first-spike latency — the **Spiking Temporal Critic (STC)**.

---

## Overview

Biological neural circuits encode information not only through spike _rate_ but also through precise spike _timing_. This repository investigates whether that temporal coding principle can improve reinforcement learning by replacing a conventional ANN value critic with an SNN whose output neuron fires earlier for high-value states and later (or not at all) for low-value states.

### Key Components

| Module                | Description                                                                     |
| --------------------- | ------------------------------------------------------------------------------- |
| `SNNTimingCritic`     | Timing-based SNN critic — value is the soft first-spike time mapped to a scalar |
| `SNNSpikeActor`       | Spike-count SNN actor with Poisson input encoding and potential fallback        |
| `SNNSpikeValueCritic` | Rate-coded SNN critic (baseline for ANN→SNN conversion)                         |
| `ActorCritic`         | Generic wrapper supporting critic-informed actor variants                       |

### Experiment Configurations

| Mode                   | Actor             | Critic        | Config key                    |
| ---------------------- | ----------------- | ------------- | ----------------------------- |
| ANN Baseline           | ANN MLP           | ANN MLP       | `ann_baseline`                |
| SNN Actor + ANN Critic | SNN (spike-count) | ANN MLP       | `snn_actor_ann_critic`        |
| **STC (ours)**         | SNN (spike-count) | SNN (timing)  | `snn_actor_snn_timing_critic` |
| ANN→SNN Actor          | Converted SNN     | ANN MLP       | `ann2snn_actor`               |
| ANN→SNN Full           | Converted SNN     | Converted SNN | `ann2snn_both`                |

---

## Repository Structure

```
.
├── configs/                        # Per-environment YAML configs
│   ├── cartpole/
│   ├── tmaze/
│   ├── poc/                        # Proof-of-concept (partial CartPole)
│   └── FetchReachDense-v4/
├── experiments/                    # Training entry-point scripts
├── post_analysis/                  # Plotting, benchmarking & statistical tests
├── scripts/                        # Ablation runners and helper scripts
├── src/
│   ├── envs/                       # Custom environments (T-Maze)
│   ├── models/                     # Neural network architectures
│   │   ├── actor_critic.py         # Generic actor-critic wrapper
│   │   ├── ann.py                  # ANN backbone, actor & critic heads
│   │   ├── recurrent_ann.py        # LSTM-based recurrent backbone
│   │   ├── snn_spike_actor.py      # Spike-count SNN actor
│   │   ├── snn_spike_value_critic.py  # Rate-coded SNN critic
│   │   ├── snn_timing_critic.py    # Timing-coded SNN critic (STC)
│   │   ├── snn_block.py            # Reusable SNN layer (Linear + LIF)
│   │   ├── snn_utils.py            # Poisson encoding & LIF state helpers
│   │   └── surrogates.py           # Custom surrogate gradient functions
│   ├── training/                   # PPO training loop & utilities
│   │   ├── agents.py               # Agent factory (make_agent)
│   │   ├── core_trainer.py         # Main PPO training loop
│   │   ├── baseline_trainer.py     # ANN baseline entry point
│   │   ├── surrogate_trainer.py    # End-to-end SNN PPO trainer
│   │   ├── conversion_trainer.py   # ANN→SNN conversion pipeline
│   │   ├── gae.py                  # Rollout collection & GAE
│   │   ├── ppo_update.py           # Clipped PPO update step
│   │   ├── evaluate.py             # Evaluation protocols
│   │   ├── envs.py                 # Environment wrappers
│   │   ├── hooks.py                # Training callbacks (energy monitoring)
│   │   └── record.py               # Video recording utilities
│   ├── conversion/                 # ANN-to-SNN conversion utilities
│   └── utils/                      # Logging, plotting, checkpointing
├── run_experiments.sh              # Multi-seed experiment runner
├── submit_all.sh                   # SLURM batch submission helper
├── submit_array.slurm              # SLURM array job template
└── requirements.txt
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU (recommended; CPU supported but slow)
- [conda](https://docs.conda.io/) or [venv](https://docs.python.org/3/library/venv.html)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Samuel-Mbah/spiking-temporal-critic-stc.git
cd spiking-temporal-critic-stc

# 2. Create and activate a virtual environment
conda create -n stc python=3.11 -y
conda activate stc

# 3. Install dependencies
pip install -r requirements.txt
```

> **Weights & Biases (optional):** Set `reporting.wandb.use: true` in your config and run `wandb login` to enable experiment tracking.

---

## Quick Start

### Training the STC agent on CartPole

```bash
python experiments/snn_actor_snn_timing_critic.py \
    --config configs/cartpole/snn_actor_snn_timing_critic.yaml \
    --seed 1 \
    --run-name seed_1 \
    --log-dir results/logs/snn_actor_snn_timing_critic/seed_1
```

### Training the ANN Baseline

```bash
python experiments/ann_baseline.py \
    --config configs/cartpole/ann_baseline.yaml \
    --seed 1 \
    --run-name seed_1 \
    --log-dir results/logs/ann_baseline/seed_1
```

### Running All Experiments (multiple seeds)

```bash
# CartPole / proof-of-concept suite (5 seeds each)
SUITE=poc SEED_COUNT=5 ./run_experiments.sh

# T-Maze suite — passive variant
SUITE=tmaze TMAZE_ACTIVE=false SEED_COUNT=5 ./run_experiments.sh

# T-Maze suite — active variant
SUITE=tmaze TMAZE_ACTIVE=true SEED_COUNT=5 ./run_experiments.sh
```

### SLURM Cluster Submission

```bash
# Submit all experiments as SLURM array jobs (seeds 1–5)
./submit_all.sh
```

---

## Environments

| Environment        | ID                          | Description                                  |
| ------------------ | --------------------------- | -------------------------------------------- |
| CartPole-v1        | `CartPole-v1`               | Classic balancing task                       |
| T-Maze (passive)   | `tmaze-v0`                  | Memory-dependent navigation, fixed cue       |
| T-Maze (active)    | `tmaze-v0` (active=true)    | Memory-dependent navigation, distractor cues |
| FetchReachDense-v4 | `FetchReachDense-v4`        | Robotic arm reaching (continuous)            |
| Proof-of-concept   | `CartPole-v1` (partial obs) | Partial-observation variant of CartPole      |

---

## Models

### `SNNTimingCritic` (the STC critic)

Encodes the value of a state via **first-spike latency**:

- Input observations are (optionally) Poisson-encoded into spike trains.
- A 2-layer LIF network processes spikes for `T` timesteps.
- The soft first-spike time `τ ∈ [0, T)` is computed via a differentiable sigmoid approximation.
- `τ` is mapped linearly to a scalar value in `[Rmin, Rmax]`: early spikes → high value.

Key hyperparameters (`snn` section of config):

| Parameter           | Default | Description                             |
| ------------------- | ------- | --------------------------------------- |
| `critic_T`          | `32`    | Simulation window length                |
| `Rmax`              | `500.0` | Value assigned to earliest spike        |
| `Rmin`              | `0.0`   | Value assigned to latest / no spike     |
| `critic_cosh_alpha` | `10.0`  | Cosh surrogate gradient scale           |
| `critic_spike_temp` | `25.0`  | Sigmoid temperature for soft spike-time |

### `SNNSpikeActor`

Encodes policy logits via **accumulated synaptic current** over `T` steps:

- Optional Poisson input encoding.
- Logit centering and temperature scaling for stable entropy.
- Potential fallback for zero-spike frames prevents degenerate behaviour.

### `ActorCritic` wrapper

Supports _critic-informed actor_ mode where the critic's value estimate is
normalised and injected as an additional input to the actor, enabling
value-modulated policy updates.

---

## Post-Analysis

After training, generate publication-quality figures with the scripts in
`post_analysis/`:

```bash
# Aggregated reward curves across all experiments and seeds
python post_analysis/plot_all_comparisons.py

# Spike-activity figures
python post_analysis/generate_spike_activity_figure.py

# Latency & energy tables (T-Maze)
python post_analysis/generate_tmaze_latency_energy_tables.py

# Timing-critic mechanistic figures (T-Maze)
python post_analysis/generate_tmaze_timing_mechanistic_figures.py \
    --config configs/tmaze/snn_actor_snn_timing_critic.yaml

# Statistical paired-seed significance tests
python post_analysis/paired_seed_stats.py \
    --model-a-dir results/logs/ann_baseline \
    --model-b-dir results/logs/snn_actor_snn_timing_critic
```

---

## Configuration Reference

All experiments are controlled by YAML config files in `configs/`. Key sections:

```yaml
env: # Environment ID, number of parallel envs, frame-stacking
training: # Rollout length, total updates, batch size, update epochs
ppo: # Learning rate, clip epsilon, entropy coefficient, KL target
model: # Architecture mode, hidden dim, critic_informs_actor flag
snn: # SNN-specific: T, beta, V_th, surrogate params, Rmax/Rmin
benchmark: # Inference energy measurement settings
reporting: # Plot saving, W&B integration
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mbah2024stc,
  title   = {Spiking Temporal Critic: Latency-Coded Value Learning In Spiking Actor-Critc Reinforcement Learning},
  author  = {Mbah, Samuel},
  year    = {2024},
  url     = {https://github.com/Samuel-Mbah/spiking-temporal-critic-stc}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
