#!/bin/bash
SESSION="neuroai"
CONDA_ENV="SNNAC" # <--- Change this to your environment name

# Start session
tmux new-session -d -s $SESSION

# Pane 1: Activate and Run Experiment
tmux send-keys -t $SESSION "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION "sleep 1; python3 experiments/snn_actor_snn_timing_critic.py --config configs/snn_actor_snn_timing_critic.yaml" C-m

# Split vertically
tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION "conda activate $CONDA_ENV" C-m

# Pane 2: GPU Monitoring
tmux send-keys -t $SESSION "watch -n 1 nvidia-smi" C-m

# Split horizontally
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "conda activate $CONDA_ENV" C-m

# Pane 3: Watch Logs
# Suggestion: Using 'tail -f' on your most recent log file
tmux send-keys -t $SESSION "tail -f $(ls -t logs/*.log | head -n 1)" C-m

# Focus back on the first pane and attach
tmux select-pane -t 0
tmux attach-session -t $SESSION
