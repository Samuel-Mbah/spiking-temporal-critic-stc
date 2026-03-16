
import numpy as np
from typing import Dict, Any, Tuple

from src.training.gae import collect_rollout
from src.training.ppo_update import update_policy
from src.training.evaluate import evaluate, get_spike_stats_safe

class CoreTrainer:
    """
    Orchestrates the PPO training loop.
    Decouples environment interaction (collect_rollout) from optimization (update_policy).
    """
    def __init__(
        self,
        agent,
        env_train,
        env_eval,
        optimizer,
        logger,
        config: Dict[str, Any],
        hooks: Dict[str, Any] = None,
    ):
        self.agent = agent
        self.env_train = env_train
        self.env_eval = env_eval
        self.optimizer = optimizer
        self.logger = logger
        self.cfg = config
        self.hooks = hooks or {}
        self._prev_total_spikes = None
        self._prev_total_spike_timesteps = None
        self._prev_actor_spikes = None
        self._prev_actor_spike_timesteps = None
        self._prev_critic_spikes = None
        self._prev_critic_spike_timesteps = None

    def train_episode(self) -> Tuple[float, Dict[str, float]]:
        """
        Executes one PPO iteration:
        1. Collects a rollout trajectory.
        2. Computes advantages (GAE).
        3. Updates policy via PPO.
        """
        if self.hooks.get("on_rollout_start"):
            self.hooks["on_rollout_start"](self.agent)

        train_cfg = self.cfg.get("training", {})
        ppo_cfg = self.cfg.get("ppo", {})
        sticky_cfg = self.cfg.get("sticky_action", {})
        sticky = sticky_cfg.get("train", train_cfg.get("sticky_action", False))

        # 1. Collect Rollout Data
        states, actions, logp_old, adv, ret, values_old, raw_reward_mean, raw_episode_count, done_count = collect_rollout(
            self.env_train,
            self.agent,
            gamma=ppo_cfg.get("gamma", 0.99),
            lam=ppo_cfg.get("lam", 0.95),
            sticky_action=sticky,
            n_steps=train_cfg.get("rollout_length", 2048),
            no_action_cfg=self.cfg.get("snn", {}).get("no_action", {})
        )
        
        # 2. Extract & Log Spike Statistics
        if hasattr(self.agent, 'get_spike_stats') or hasattr(self.agent, "actor"):
            # Pull cumulative stats separately from actor and critic modules.
            actor_stats = get_spike_stats_safe(getattr(self.agent, "actor", None))
            critic_stats = get_spike_stats_safe(getattr(self.agent, "critic", None))

            actor_spikes_cum = float(actor_stats.get("total_spikes", 0.0))
            actor_timesteps_cum = float(actor_stats.get("total_timesteps", 0.0))
            critic_spikes_cum = float(critic_stats.get("total_spikes", 0.0))
            critic_timesteps_cum = float(critic_stats.get("total_timesteps", 0.0))

            total_spikes_cum = actor_spikes_cum + critic_spikes_cum
            total_spike_timesteps_cum = actor_timesteps_cum + critic_timesteps_cum

            if self._prev_total_spikes is None or total_spikes_cum < self._prev_total_spikes:
                rollout_spikes = total_spikes_cum
            else:
                rollout_spikes = total_spikes_cum - self._prev_total_spikes

            if self._prev_actor_spikes is None or actor_spikes_cum < self._prev_actor_spikes:
                rollout_actor_spikes = actor_spikes_cum
            else:
                rollout_actor_spikes = actor_spikes_cum - self._prev_actor_spikes

            if self._prev_critic_spikes is None or critic_spikes_cum < self._prev_critic_spikes:
                rollout_critic_spikes = critic_spikes_cum
            else:
                rollout_critic_spikes = critic_spikes_cum - self._prev_critic_spikes

            if (
                self._prev_total_spike_timesteps is None
                or total_spike_timesteps_cum < self._prev_total_spike_timesteps
            ):
                rollout_spike_timesteps = total_spike_timesteps_cum
            else:
                rollout_spike_timesteps = (
                    total_spike_timesteps_cum - self._prev_total_spike_timesteps
                )

            if (
                self._prev_actor_spike_timesteps is None
                or actor_timesteps_cum < self._prev_actor_spike_timesteps
            ):
                rollout_actor_timesteps = actor_timesteps_cum
            else:
                rollout_actor_timesteps = actor_timesteps_cum - self._prev_actor_spike_timesteps

            if (
                self._prev_critic_spike_timesteps is None
                or critic_timesteps_cum < self._prev_critic_spike_timesteps
            ):
                rollout_critic_timesteps = critic_timesteps_cum
            else:
                rollout_critic_timesteps = critic_timesteps_cum - self._prev_critic_spike_timesteps

            # Log rollout and cumulative variants explicitly.
            self.logger.record("spikes/total", rollout_spikes)
            self.logger.record("spike_count_total", rollout_spikes)
            self.logger.record("spikes/cumulative_total", total_spikes_cum)
            self.logger.record("spikes/actor", rollout_actor_spikes)
            self.logger.record("spikes/critic", rollout_critic_spikes)
            self.logger.record("spikes/cumulative_actor", actor_spikes_cum)
            self.logger.record("spikes/cumulative_critic", critic_spikes_cum)

            if rollout_spike_timesteps is not None and rollout_spike_timesteps > 0:
                firing_rate = max(0.0, min(1.0, rollout_spikes / rollout_spike_timesteps))
                sparsity = 1.0 - firing_rate
                self.logger.record("spikes/firing_rate", firing_rate)
                self.logger.record("spikes/sparsity", sparsity)
            else:
                sparsity = actor_stats.get("sparsity", None)
                if sparsity is not None:
                    self.logger.record("spikes/sparsity", sparsity)
                    firing_rate = max(0.0, min(1.0, 1.0 - float(sparsity)))
                    self.logger.record("spikes/firing_rate", firing_rate)

            steps_in_rollout = train_cfg.get("rollout_length", 2048) * getattr(self.env_train, "num_envs", 1)
            if steps_in_rollout > 0:
                self.logger.record("spikes/per_step", float(rollout_spikes) / float(steps_in_rollout))
                self.logger.record("spikes/actor_per_step", float(rollout_actor_spikes) / float(steps_in_rollout))
                self.logger.record("spikes/critic_per_step", float(rollout_critic_spikes) / float(steps_in_rollout))
            if rollout_actor_timesteps > 0:
                self.logger.record("spikes/actor_firing_rate", float(rollout_actor_spikes) / float(rollout_actor_timesteps))
            if rollout_critic_timesteps > 0:
                self.logger.record("spikes/critic_firing_rate", float(rollout_critic_spikes) / float(rollout_critic_timesteps))
            spike_latency = actor_stats.get("mean_latency", None)
            if spike_latency is not None:
                self.logger.record("latency/spike_timing_steps", float(spike_latency))

            self._prev_total_spikes = total_spikes_cum
            self._prev_total_spike_timesteps = total_spike_timesteps_cum
            self._prev_actor_spikes = actor_spikes_cum
            self._prev_actor_spike_timesteps = actor_timesteps_cum
            self._prev_critic_spikes = critic_spikes_cum
            self._prev_critic_spike_timesteps = critic_timesteps_cum

        # 3. Determine Reportable Reward
        # If raw_reward_mean is valid (episodes finished), use it. 
        # Otherwise, fallback to normalized return mean to avoid empty plots, but prefer raw.
        if not np.isnan(raw_reward_mean):
            tr = raw_reward_mean
        else:
            tr = ret.mean().item()

        # Log raw (unscaled) rollout reward for plotting when available.
        raw_logged = float(raw_reward_mean) if not np.isnan(raw_reward_mean) else np.nan
        self.logger.record("train/rollout_reward_raw", raw_logged)
        
        self.logger.record("train/rollout_episode_count", raw_episode_count)
        self.logger.record("train/rollout_done_count", done_count)
            
        self.logger.num_timesteps += len(states)

        if self.hooks.get("on_rollout_end"):
            self.hooks["on_rollout_end"](self.agent, tr)

        # 4. PPO Update
        spike_reg_cfg = self.cfg.get("spike_reg", {})
        update_kwargs = {
            **ppo_cfg, 
            "batch_size": train_cfg.get("batch_size", 256),
            "n_epochs": train_cfg.get("update_epochs", 10),
            "shuffle_minibatches": train_cfg.get("shuffle_minibatches", True),
            "lambda_spike": spike_reg_cfg.get("lambda_spike", ppo_cfg.get("lambda_spike", 1e-3)),
        }

        metrics = update_policy(
            self.agent,
            states,
            actions,
            logp_old,
            adv,
            ret,
            self.optimizer,
            values_old=values_old,
            logger=self.logger,
            **update_kwargs
        )

        return tr, metrics

    def evaluate(self) -> Tuple[float, float, float, int, int]:
        """
        Runs evaluation episodes.
        Returns: (Average Reward, Success Rate %, Average Length, Success Count, Num Episodes)
        """
        if self.hooks.get("on_eval_start"):
            self.hooks["on_eval_start"](self.agent)

        n_episodes = int(self.cfg.get("ppo", {}).get("eval_episodes", 5))
        train_cfg = self.cfg.get("training", {})
        sticky_cfg = self.cfg.get("sticky_action", {})
        sticky = sticky_cfg.get("eval", train_cfg.get("sticky_action", False))
        threshold = float(self.cfg.get("ppo", {}).get("reward_threshold", 475.0))

        rewards = []
        lengths = []
        latencies = []
        wall_clock = []
        eval_actor_spikes = []
        eval_critic_spikes = []
        eval_total_spikes = []
        success_count = 0
        eval_seed_base = self.cfg.get("ppo", {}).get("eval_seed")

        for _ in range(n_episodes):
            seed = None
            if eval_seed_base is not None:
                seed = int(eval_seed_base) + len(rewards)
            # Pass return_metrics=True to capture latency/spikes
            ep_reward, ep_steps, metrics = evaluate(
                self.env_eval,
                self.agent,
                sticky_action=sticky,
                return_metrics=True,
                seed=seed,
            )
            rewards.append(ep_reward)
            lengths.append(ep_steps)
            
            # Collect latency if available
            if "eval/latency" in metrics:
                latencies.append(metrics["eval/latency"])
            if "eval/wall_clock_ms" in metrics:
                wall_clock.append(metrics["eval/wall_clock_ms"])
            if "eval/spikes_actor" in metrics:
                eval_actor_spikes.append(metrics["eval/spikes_actor"])
            if "eval/spikes_critic" in metrics:
                eval_critic_spikes.append(metrics["eval/spikes_critic"])
            if "eval/spikes" in metrics:
                eval_total_spikes.append(metrics["eval/spikes"])

            if ep_reward >= threshold:
                success_count += 1

        avg_r = np.mean(rewards) if rewards else 0.0
        avg_l = np.mean(lengths) if lengths else 0.0
        success_rate = (success_count / max(1, n_episodes)) * 100.0
        avg_latency = np.mean(latencies) if latencies else 0.0
        avg_wall_clock = np.mean(wall_clock) if wall_clock else 0.0

        # Log Evaluation Metrics
        self.logger.record("eval/latency", avg_latency)
        self.logger.record("latency/eval_spike_timing_steps", avg_latency)
        if avg_wall_clock > 0:
            self.logger.record("latency/eval_wall_clock_ms", avg_wall_clock)
        self.logger.record("eval/ep_len", avg_l)
        if eval_total_spikes:
            avg_eval_spikes = float(np.mean(eval_total_spikes))
            self.logger.record("eval/spikes", avg_eval_spikes)
            self.logger.record("spikes/eval_total", avg_eval_spikes)
            self.logger.record("eval/spikes_per_step", avg_eval_spikes / avg_l if avg_l > 0 else 0.0)
        if eval_actor_spikes:
            avg_eval_actor = float(np.mean(eval_actor_spikes))
            self.logger.record("eval/spikes_actor", avg_eval_actor)
            self.logger.record("spikes/eval_actor_total", avg_eval_actor)
            self.logger.record("eval/spikes_actor_per_step", avg_eval_actor / avg_l if avg_l > 0 else 0.0)
        if eval_critic_spikes:
            avg_eval_critic = float(np.mean(eval_critic_spikes))
            self.logger.record("eval/spikes_critic", avg_eval_critic)
            self.logger.record("spikes/eval_critic_total", avg_eval_critic)
            self.logger.record("eval/spikes_critic_per_step", avg_eval_critic / avg_l if avg_l > 0 else 0.0)

        if self.hooks.get("on_eval_end"):
            self.hooks["on_eval_end"](self.agent, avg_r)

        return avg_r, success_rate, avg_l, success_count, n_episodes
























