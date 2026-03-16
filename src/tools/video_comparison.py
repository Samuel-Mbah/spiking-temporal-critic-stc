import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import shutil
import subprocess
import tempfile
from matplotlib.backends.backend_agg import FigureCanvasAgg
import gymnasium as gym

from src.training.evaluate import get_model_device, get_last_spike_count
from src.training.envs import apply_obs_wrappers

class LayerHook:
    """Safely captures intermediate activations from a PyTorch module."""
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = []

    def hook_fn(self, module, input, output):
        # snn.Leaky returns a tuple (spikes, mem), Linear returns a tensor
        out = output[0] if isinstance(output, tuple) else output
        self.features.append(out.detach().cpu().numpy())

    def get_and_clear(self):
        if not self.features:
            return np.zeros(32) # fallback shape
        # SNNs will have T items per env step, ANNs will have 1. Average them.
        res = np.mean(self.features, axis=0).flatten()
        self.features = []
        return res

    def close(self):
        self.hook.remove()

def get_target_layer(agent, is_snn):
    """Finds the first hidden layer to visualize."""
    if is_snn:
        # Capture the current (i1) entering the first LIF neuron layer
        if hasattr(agent.actor, "block1"):
            return agent.actor.block1.linear
    else:
        # Capture the first hidden dense layer in the standard ANN
        for m in agent.actor.modules():
            if isinstance(m, nn.Linear):
                return m
    return agent.actor # Fallback

@torch.no_grad()
def collect_trajectory(agent, env_id, seed=42, is_snn=False, max_steps=500, **env_kwargs):
    """
    Runs the agent and collects step-by-step RGB frames, critic values, 
    and internal neural states (via hooks).
    """
    agent.eval()
    device = get_model_device(agent)

    # Attach hook to capture internal dynamics
    target_layer = get_target_layer(agent, is_snn)
    hook = LayerHook(target_layer)

    # Setup Environment
    env = gym.make(env_id, render_mode="rgb_array", **env_kwargs.get("env_kwargs", {}))
    env = apply_obs_wrappers(
        env, 
        partial_obs=env_kwargs.get("partial_obs"), 
        frame_stack=env_kwargs.get("frame_stack"), 
        frame_stack_flatten=env_kwargs.get("frame_stack_flatten", True),
        pad_video_tail=env_kwargs.get("pad_video_tail", False),
    )
    
    obs, _ = env.reset(seed=seed)
    
    data = {
        "frames": [],
        "values": [],
        "internal_states": [], 
        "actions": [],
        "critic_vout_traj": [],
        "rewards": [],
        "terminated": False,
        "truncated": False,
        "steps": 0,
        "return": 0.0,
    }

    prev_action = None
    
    for _ in range(max_steps):
        data["frames"].append(env.render())

        # 1. Normalization (Matches evaluate.py)
        if hasattr(agent, "obs_rms"):
            rms = agent.obs_rms
            obs = np.clip((obs - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        # 2. Forward Pass (Matches ActorCritic.py)
        # Returns (logits, value)
        logits, val_tensor = agent(obs_t) 
        
        data["values"].append(val_tensor.item() if val_tensor is not None else 0.0)
        data["internal_states"].append(hook.get_and_clear())
        # If available (SNN timing critic), store raw internal output-membrane trajectory [T].
        vout = getattr(getattr(agent, "critic", None), "_last_vout", None)
        if isinstance(vout, torch.Tensor):
            if vout.ndim == 3 and vout.shape[1] > 0 and vout.shape[2] > 0:
                data["critic_vout_traj"].append(vout[:, 0, 0].detach().cpu().numpy())
            else:
                data["critic_vout_traj"].append(None)
        else:
            data["critic_vout_traj"].append(None)

        # 3. Action Selection
        action = int(torch.argmax(logits, dim=-1).item())

        # 4. Sticky Action Logic (SNNs only)
        allow_sticky = is_snn and hasattr(agent, "actor")
        if allow_sticky:
            if get_last_spike_count(agent.actor) == 0.0 and prev_action is not None:
                action = prev_action

        data["actions"].append(action)

        # 5. Step Environment
        obs, reward, terminated, truncated, _ = env.step(action)
        data["rewards"].append(float(reward))
        prev_action = action

        if terminated or truncated:
            data["frames"].append(env.render())
            data["terminated"] = bool(terminated)
            data["truncated"] = bool(truncated)
            break

    data["steps"] = len(data["actions"])
    data["return"] = float(np.sum(data["rewards"])) if data["rewards"] else 0.0

    hook.close()
    env.close()
    return data

def render_comparison_video(
    ann_data,
    snn_data,
    output_path="comparison.mp4",
    fps=5,
    value_ylim=None,
    snn_critic_T=32,
    show_raw_critic_vout=False,
    end_hold_frames=20,
    metadata=None,
    use_production_codec=True,
    crf=20,
):
    """
    Stitches trajectory data into a synchronized multi-panel MP4 using matplotlib.
    """
    # Work on local copies so repeated renders (main + montage) are deterministic.
    ann_data = {
        **ann_data,
        "frames": list(ann_data.get("frames", [])),
        "values": list(ann_data.get("values", [])),
        "internal_states": list(ann_data.get("internal_states", [])),
        "actions": list(ann_data.get("actions", [])),
        "critic_vout_traj": list(ann_data.get("critic_vout_traj", [])),
    }
    snn_data = {
        **snn_data,
        "frames": list(snn_data.get("frames", [])),
        "values": list(snn_data.get("values", [])),
        "internal_states": list(snn_data.get("internal_states", [])),
        "actions": list(snn_data.get("actions", [])),
        "critic_vout_traj": list(snn_data.get("critic_vout_traj", [])),
    }

    # Save true lengths before any plotting-only padding.
    ann_true_len = len(ann_data["actions"])
    snn_true_len = len(snn_data["actions"])

    # Hold the final frame/value for readability without affecting env-step metrics.
    hold = max(0, int(end_hold_frames))
    if hold > 0:
        for data in [ann_data, snn_data]:
            if data["frames"]:
                for _ in range(hold):
                    data["frames"].append(data["frames"][-1])
                    if data["values"]:
                        data["values"].append(data["values"][-1])
                    if data["internal_states"]:
                        data["internal_states"].append(data["internal_states"][-1])

    # Pad shorter trajectory with its last frame to match lengths
    max_len = max(len(ann_data["frames"]), len(snn_data["frames"]))
    for data in [ann_data, snn_data]:
        while len(data["frames"]) < max_len:
            data["frames"].append(data["frames"][-1])
            data["values"].append(data["values"][-1])
            data["internal_states"].append(data["internal_states"][-1])

    # Setup OpenCV Video Writer
    fig_w, fig_h = 16, 14
    video_size = (fig_w * 100, fig_h * 100) # 100 dpi
    tmp_out = output_path
    if use_production_codec and shutil.which("ffmpeg"):
        base = os.path.dirname(output_path) or "."
        os.makedirs(base, exist_ok=True)
        tmp_out = tempfile.NamedTemporaryFile(prefix="tmp_raw_", suffix=".mp4", dir=base, delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_out, fourcc, fps, video_size)

    plt.style.use("dark_background")

    # Determine value axis range.
    all_values = []
    all_values.extend(ann_data.get("values", []))
    all_values.extend(snn_data.get("values", []))
    if value_ylim is not None:
        y_min, y_max = float(value_ylim[0]), float(value_ylim[1])
    elif all_values:
        v_min = float(np.min(all_values))
        v_max = float(np.max(all_values))
        pad = max(1e-3, 0.05 * max(abs(v_min), abs(v_max), v_max - v_min, 1.0))
        y_min, y_max = v_min - pad, v_max + pad
    else:
        y_min, y_max = 0.0, 1.0
    
    for t in range(max_len):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
        if metadata:
            meta_text = " | ".join([f"{k}: {v}" for k, v in metadata.items() if v is not None])
            fig.text(
                0.01,
                0.99,
                meta_text,
                fontsize=12,
                color="white",
                va="top",
                ha="left",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.3"),
            )
        
        # --- TOP ROW: Environment Renders ---
        ax_env_ann = plt.subplot2grid((5, 2), (0, 0), rowspan=2)
        ax_env_ann.imshow(ann_data["frames"][t])
        ax_env_ann.set_title(f"ANN Agent (t={t})", fontsize=18, fontweight="bold")
        ax_env_ann.axis("off")

        ax_env_snn = plt.subplot2grid((5, 2), (0, 1), rowspan=2)
        ax_env_snn.imshow(snn_data["frames"][t])
        ax_env_snn.set_title(f"SNN Agent (t={t})", fontsize=18, fontweight="bold")
        ax_env_snn.axis("off")

        # --- MIDDLE ROW: Internal Dynamics Heatmaps ---
        ax_dyn_ann = plt.subplot2grid((5, 2), (2, 0))
        ann_state_history = np.array(ann_data["internal_states"][:t+1])
        if ann_state_history.size > 0:
            ax_dyn_ann.imshow(ann_state_history.T, aspect='auto', cmap='magma', origin='lower')
        ax_dyn_ann.set_title("ANN Layer 1 Activations", fontsize=14)
        ax_dyn_ann.set_ylabel("Neuron Index", fontsize=13)
        ax_dyn_ann.tick_params(axis="both", labelsize=11)

        ax_dyn_snn = plt.subplot2grid((5, 2), (2, 1))
        snn_state_history = np.array(snn_data["internal_states"][:t+1])
        if snn_state_history.size > 0:
            ax_dyn_snn.imshow(snn_state_history.T, aspect='auto', cmap='viridis', origin='lower')
        ax_dyn_snn.set_title("SNN Layer 1 Mean Input Current ($I_{in}$)", fontsize=14)
        ax_dyn_snn.set_ylabel("Neuron Index", fontsize=13)
        ax_dyn_snn.tick_params(axis="both", labelsize=11)

        # --- BOTTOM ROW: RL Critic Value Tracker ---
        ax_val_ann = plt.subplot2grid((5, 2), (3, 0))
        ann_vals = ann_data["values"][:min(t + 1, ann_true_len)]
        ann_x = np.arange(len(ann_vals))
        ax_val_ann.plot(ann_x, ann_vals, color='cyan', lw=2, marker='o', markersize=4)
        ax_val_ann.set_xlim(0, max_len)
        ax_val_ann.set_ylim(y_min, y_max)
        ax_val_ann.set_title("ANN Critic Value by Env Step", fontsize=14)
        ax_val_ann.set_xlabel("Env Step (Decision Step)", fontsize=14)
        ax_val_ann.tick_params(axis="both", labelsize=11)
        if ann_vals:
            ax_val_ann.text(
                0.02,
                0.92,
                f"V={ann_vals[-1]:.3f}",
                transform=ax_val_ann.transAxes,
                color="cyan",
                fontsize=13,
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.2"),
            )

        ax_val_snn = plt.subplot2grid((5, 2), (3, 1))
        snn_vals = snn_data["values"][:min(t + 1, snn_true_len)]
        snn_x = np.arange(len(snn_vals))
        ax_val_snn.plot(snn_x, snn_vals, color='lime', lw=2, marker='o', markersize=4)
        ax_val_snn.set_xlim(0, max_len)
        ax_val_snn.set_ylim(y_min, y_max)
        ax_val_snn.set_title("SNN Critic Value by Env Step", fontsize=14)
        ax_val_snn.set_xlabel("Env Step (Decision Step)", fontsize=14)
        ax_val_snn.tick_params(axis="both", labelsize=11)
        if snn_vals:
            ax_val_snn.text(
                0.02,
                0.92,
                f"V={snn_vals[-1]:.3f}",
                transform=ax_val_snn.transAxes,
                color="lime",
                fontsize=13,
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.2"),
            )

        # --- NEW ROW: SNN critic internal timing (micro-steps) ---
        ax_int_ann = plt.subplot2grid((5, 2), (4, 0))
        ax_int_ann.axis("off")
        ax_int_ann.text(
            0.02,
            0.45,
            "ANN has no internal critic timing loop.\n(1 value per env decision step)",
            color="white",
            fontsize=12,
            ha="left",
            va="center",
        )

        ax_int_snn = plt.subplot2grid((5, 2), (4, 1))
        n_snn = len(snn_vals)
        if n_snn > 0:
            critic_T = max(1, int(snn_critic_T))
            internal_x = np.arange(n_snn * critic_T)
            internal_vals = np.repeat(np.asarray(snn_vals, dtype=np.float32), critic_T)
            raw_trajs = snn_data.get("critic_vout_traj", [])
            use_raw = bool(show_raw_critic_vout) and len(raw_trajs) >= n_snn and all(
                isinstance(raw_trajs[i], np.ndarray) and raw_trajs[i].size == critic_T for i in range(n_snn)
            )
            if use_raw:
                internal_vals = np.concatenate([raw_trajs[i].astype(np.float32) for i in range(n_snn)], axis=0)
                ax_int_snn.plot(internal_x, internal_vals, color="#ff9f1c", lw=1.5, label="raw vout(t)")
                ax_int_snn.plot(
                    internal_x,
                    np.repeat(np.asarray(snn_vals, dtype=np.float32), critic_T),
                    color="#ffd166",
                    lw=1.8,
                    alpha=0.8,
                    label="step value (repeat)",
                )
                ax_int_snn.legend(loc="lower right", fontsize=9, framealpha=0.5)
            else:
                ax_int_snn.plot(internal_x, internal_vals, color="#ffd166", lw=2)
            ax_int_snn.set_xlim(0, max(1, n_snn * critic_T - 1))
            ax_int_snn.set_ylim(y_min, y_max)
            ax_int_snn.set_title("SNN Timing Critic: Internal Timestep vs Value", fontsize=14)
            ax_int_snn.set_xlabel("SNN Critic Internal Timestep", fontsize=14)
            ax_int_snn.set_ylabel("Value / vout", fontsize=13)
            ax_int_snn.tick_params(axis="both", labelsize=11)
            # Mark environment-step boundaries every critic_T internal ticks.
            for b in range(critic_T, n_snn * critic_T, critic_T):
                ax_int_snn.axvline(b, color="#8d99ae", linestyle="--", linewidth=0.9, alpha=0.7)

            current_internal_t = n_snn * critic_T - 1
            current_value = float(internal_vals[-1])
            ax_int_snn.text(
                0.02,
                0.88,
                f"critic_T={critic_T}, env_step={n_snn - 1}, internal_t={current_internal_t}, value={current_value:.3f}",
                transform=ax_int_snn.transAxes,
                color="#ffd166",
                fontsize=11,
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.2"),
            )

        plt.tight_layout()

        # Convert Matplotlib figure to OpenCV frame
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        # Matplotlib >=3.10 removed tostring_rgb() on FigureCanvasAgg.
        img = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        out.write(img)
        plt.close(fig)

    out.release()
    if use_production_codec and shutil.which("ffmpeg"):
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_out,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", str(int(crf)),
                "-movflags", "+faststart",
                output_path,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            # Fallback to the temporary raw mp4 when ffmpeg transcoding fails.
            if tmp_out != output_path and os.path.exists(tmp_out):
                os.replace(tmp_out, output_path)
    print(f"✅ Video successfully saved to {output_path}")

# if __name__ == "__main__":
    # Example Usage:
    # ann_agent = torch.load("path/to/ann.pt")
    # snn_agent = torch.load("path/to/snn.pt")
    
    # env_kwargs = {"frame_stack": 3, "frame_stack_flatten": True}
    
    # ann_data = collect_trajectory(ann_agent, env_id="TMaze-v0", is_snn=False, **env_kwargs)
    # snn_data = collect_trajectory(snn_agent, env_id="TMaze-v0", is_snn=True, **env_kwargs)
    
    # render_comparison_video(ann_data, snn_data, output_path="tmaze_comparison.mp4")
    pass
