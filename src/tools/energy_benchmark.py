"""
Energy benchmarking suite for SNN vs. ANN comparison.
"""

import time
import threading
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception as e:
    logger.warning(f"GPU energy monitoring disabled. (Error: {e})")
    pynvml = None
    _NVML = False


# =======================================================================
#  HELPER: RECURSIVE STATS RETRIEVAL
# =======================================================================
def get_spike_stats_safe(module):
    if module is None: return {}
    if hasattr(module, "get_spike_stats"): return module.get_spike_stats()
    if hasattr(module, "actor"): return get_spike_stats_safe(module.actor)
    if hasattr(module, "backbone"): return get_spike_stats_safe(module.backbone)
    return {}

def reset_snn_stats(model):
    for m in model.modules():
        if hasattr(m, "reset_stats"): m.reset_stats()

def get_cumulative_spikes(model):
    total = 0.0
    total_elements = 0.0
    found = False
    for m in model.modules():
        if hasattr(m, "total_spikes") and hasattr(m, "total_timesteps"):
            ts = m.total_spikes
            tt = m.total_timesteps
            if isinstance(ts, torch.Tensor): ts = ts.detach().cpu().item()
            if isinstance(tt, torch.Tensor): tt = tt.detach().cpu().item()
            total += ts
            total_elements += tt
            found = True
    
    if not found: return 0.0, 0.0
    
    # Calculate true sparsity
    density = total / total_elements if total_elements > 0 else 0.0
    sparsity = 1.0 - density
    
    return total, sparsity


# =======================================================================
#  PART 1: LOW-LEVEL GPU ENERGY METER
# =======================================================================
class GPUEnergyMeter:
    def __init__(self, sample_interval: float = 0.01, gpu_index: int = 0):
        self.sample_interval = sample_interval
        self._stop = False
        self.samples: List[tuple[float, Optional[float]]] = []
        self.thread: Optional[threading.Thread] = None
        self.gpu_handle = None
        self.gpu_index = gpu_index

        if _NVML and pynvml is not None and torch.cuda.is_available():
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception as e:
                logger.warning(f"Could not get GPU handle: {e}")
                self.gpu_handle = None
        
    def _loop(self):
        while not self._stop:
            ts = time.time()
            power_watts = None
            if self.gpu_handle and pynvml is not None:
                try:
                    power_milliwatts = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                    power_watts = power_milliwatts / 1000.0
                except Exception:
                    power_watts = None
            self.samples.append((ts, power_watts))
            time.sleep(self.sample_interval)

    def start(self):
        self._stop = False
        self.samples = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, float]:
        self._stop = True
        if self.thread:
            self.thread.join()
            self.thread = None

        total_energy = 0.0
        valid_samples = 0
        sum_power = 0.0
        
        for i in range(1, len(self.samples)):
            t0, p0 = self.samples[i-1]
            t1, _  = self.samples[i]
            if p0 is not None:
                delta_t = t1 - t0
                total_energy += p0 * delta_t
                sum_power += p0
                valid_samples += 1

        avg_power = (sum_power / valid_samples) if valid_samples > 0 else 0.0
        return {"gpu_joules": total_energy, "total_joules": total_energy, "avg_power_watts": avg_power}


# =======================================================================
#  PART 2: BENCHMARKING METRICS DATACLASS
# =======================================================================
@dataclass
class EnergyMetrics:
    train_energy_joules: float            
    inference_energy_joules: float        
    total_energy_joules: float            
    dynamic_energy_joules: float          
    energy_per_episode: float
    energy_per_reward: float
    energy_delay_product: float           
    performance_per_watt: float           
    throughput_per_watt: float            
    # NOTE: inference_joules_per_step keeps legacy behavior:
    # - ANN: dynamic joules / env step
    # - SNN: dynamic joules / (env step * T)
    inference_joules_per_step: Optional[float] = None
    raw_joules_per_env_step: Optional[float] = None
    dynamic_joules_per_env_step: Optional[float] = None
    energy_per_spike: Optional[float] = None
    sparsity_factor: Optional[float] = None
    
    avg_power_watts: float = 0.0
    idle_power_watts: float = 0.0
    peak_power_watts: float = 0.0


# =======================================================================
#  PART 3: HIGH-LEVEL BENCHMARKING FRAMEWORK
# =======================================================================
class EnergyBenchmark:
    def __init__(self):
        self.measurements: List[Dict[str, Any]] = []
        self.idle_power_watts = 0.0
        self.is_calibrated = False

    def calibrate_idle(self, duration: float = 2.0):
        if not _NVML: return
        print(f"⚡ Calibrating idle power for {duration} seconds...")
        meter = GPUEnergyMeter()
        meter.start()
        time.sleep(duration)
        data = meter.stop()
        self.idle_power_watts = data.get("avg_power_watts", 0.0)
        self.is_calibrated = True
        print(f"⚡ Baseline Idle Power: {self.idle_power_watts:.3f} W")

    def measure_episode(self,
                        model: torch.nn.Module,
                        episode_fn: Callable[[torch.nn.Module], tuple[float, int, Dict]],
                        count_spikes: bool = True,
                        warmup_runs: int = 1,
                        active_repeat: int = 1) -> Dict[str, Any]:
        
        warmup_runs = max(0, int(warmup_runs))
        active_repeat = max(1, int(active_repeat))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for _ in range(warmup_runs):
            _ = episode_fn(model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if count_spikes:
            reset_snn_stats(model)

        meter = GPUEnergyMeter(sample_interval=0.01)
        start_time = time.perf_counter()
        meter.start()

        reward = 0.0
        steps = 0
        info = {}
        for _ in range(active_repeat):
            r_i, s_i, info_i = episode_fn(model)
            reward += float(r_i)
            steps += int(s_i)
            info = info_i

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        data = meter.stop()

        episode_time = end_time - start_time
        total_energy = data["total_joules"]
        idle_energy = self.idle_power_watts * episode_time
        dynamic_energy = max(0.0, total_energy - idle_energy)
        
        spike_count = 0.0
        sparsity = None
        if count_spikes:
                spike_count, sparsity = get_cumulative_spikes(model)
                
        return {
            'reward': reward,
            'steps': steps,
            'episodes_measured': active_repeat,
            'time_seconds': episode_time,
            'energy_joules': total_energy,
            'dynamic_energy_joules': dynamic_energy,
            'power_watts': total_energy / episode_time if episode_time > 0 else 0,
            'spike_count': spike_count,
            'sparsity': sparsity,
            'info': info
        }

    def benchmark_model(self,
                        model: torch.nn.Module,
                        episode_fn: Callable[[torch.nn.Module], tuple[float, int, Dict]],
                        num_episodes: int = 100,
                        model_type: str = "SNN",
                        success_threshold: float = 475.0,
                        prev_train_energy: float = 0.0,
                        warmup_runs: int = 1,
                        active_repeat: int = 1,
                        ) -> EnergyMetrics:
        
        if not self.is_calibrated and _NVML:
            self.calibrate_idle()

        measurements = []
        
        print(f"Running {model_type} energy benchmark ({num_episodes} episodes)...")

        for i in range(num_episodes):
            m = self.measure_episode(
                model,
                episode_fn,
                count_spikes=(model_type == "SNN"),
                warmup_runs=warmup_runs,
                active_repeat=active_repeat,
            )
            measurements.append(m)
            
            if (i + 1) % (max(1, num_episodes // 5)) == 0:
                print(f"  ... {i+1}/{num_episodes}")
        
        inference_energy = sum(m['energy_joules'] for m in measurements)
        total_dynamic = sum(m['dynamic_energy_joules'] for m in measurements)
        total_time = sum(m['time_seconds'] for m in measurements)
        total_reward = sum(m['reward'] for m in measurements)
        total_env_steps = sum(m['steps'] for m in measurements)
        total_measured_episodes = sum(m.get('episodes_measured', 1) for m in measurements)
        
        avg_power = inference_energy / total_time if total_time > 0 else 0
        
        energy_per_spike = None
        avg_sparsity = None
        inference_joules_per_step = None
        raw_joules_per_env_step = None
        dynamic_joules_per_env_step = None

        if total_env_steps > 0:
            raw_joules_per_env_step = inference_energy / total_env_steps
            dynamic_joules_per_env_step = total_dynamic / total_env_steps
        
        if model_type == "SNN":
            total_spikes = sum(m['spike_count'] for m in measurements)
            if total_spikes > 0:
                energy_per_spike = inference_energy / total_spikes
            valid_sparsities = [m['sparsity'] for m in measurements if m['sparsity'] is not None]
            if valid_sparsities:
                avg_sparsity = np.mean(valid_sparsities)
            
            T = 1
            if hasattr(model, "actor") and hasattr(model.actor, "T"): T = model.actor.T
            elif hasattr(model, "T"): T = model.T
            
            total_inf_steps = total_env_steps * T
            if total_inf_steps > 0:
                inference_joules_per_step = total_dynamic / total_inf_steps
        else:
            if total_env_steps > 0:
                inference_joules_per_step = total_dynamic / total_env_steps

        return EnergyMetrics(
            train_energy_joules=prev_train_energy,
            inference_energy_joules=inference_energy,
            total_energy_joules=prev_train_energy + inference_energy,
            dynamic_energy_joules=total_dynamic,
            energy_per_episode=inference_energy / max(1, total_measured_episodes),
            energy_per_reward=inference_energy / total_reward if total_reward > 0 else float('inf'),
            inference_joules_per_step=inference_joules_per_step,
            raw_joules_per_env_step=raw_joules_per_env_step,
            dynamic_joules_per_env_step=dynamic_joules_per_env_step,
            energy_delay_product=inference_energy * total_time,
            performance_per_watt=total_reward / avg_power if avg_power > 0 else 0,
            throughput_per_watt=(total_measured_episodes/total_time)/avg_power if avg_power > 0 else 0,
            energy_per_spike=energy_per_spike,
            sparsity_factor=float(avg_sparsity) if avg_sparsity is not None else None,
            avg_power_watts=avg_power,
            idle_power_watts=self.idle_power_watts,
            peak_power_watts=max(m['power_watts'] for m in measurements) if measurements else 0
        )

    def generate_report(self, snn_metrics: EnergyMetrics, ann_metrics: EnergyMetrics) -> str:
        def pct(a, b): return ((a - b) / a * 100) if a != 0 else 0.0
        def x_factor(a, b): return (a / b) if b != 0 else 0.0

        report = f"""
=== ENERGY BENCHMARK REPORT ===
(Idle Power Calibrated: {snn_metrics.idle_power_watts:.2f} W)

1. CONSUMPTION (Total vs Dynamic)
---------------------------------
Total Energy (ANN):  {ann_metrics.total_energy_joules:.3f} J
Total Energy (SNN):  {snn_metrics.total_energy_joules:.3f} J
> Total Reduction:   {pct(ann_metrics.total_energy_joules, snn_metrics.total_energy_joules):.1f}%

Dynamic Energy (Computation Only):
  ANN: {ann_metrics.dynamic_energy_joules:.3f} J
  SNN: {snn_metrics.dynamic_energy_joules:.3f} J

2. EFFICIENCY (Per Inference)
-----------------------------
J/Inference (ANN):   {ann_metrics.inference_joules_per_step if ann_metrics.inference_joules_per_step else 0.0:.6f} J
J/Inference (SNN):   {snn_metrics.inference_joules_per_step if snn_metrics.inference_joules_per_step else 0.0:.6f} J
> Improvement:       {x_factor(ann_metrics.inference_joules_per_step, snn_metrics.inference_joules_per_step):.2f}x

3. SNN SPECIFICS
----------------
Sparsity (Inactive):            {f"{snn_metrics.sparsity_factor:.2%}" if snn_metrics.sparsity_factor else "N/A"}
Energy/Spike:        {f"{(snn_metrics.energy_per_spike*1e9):.2f} nJ" if snn_metrics.energy_per_spike else "N/A"}
"""
        # Force print to stdout in case calling script doesn't
        print(report)
        return report

if __name__ == "__main__":
    print("Energy Benchmark Module Loaded.")
