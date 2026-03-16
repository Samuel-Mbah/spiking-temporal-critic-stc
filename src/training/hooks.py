import time
import torch
import logging
from typing import Dict, Optional, Tuple

from src.tools.energy_benchmark import GPUEnergyMeter

logger = logging.getLogger(__name__)

class EnergyHook:
    """
    Lifecycle hook to measure energy during training rollouts and evaluation.
    """
    def __init__(self, sample_interval: float = 0.02, gpu_index: int = 0):
        self.sample_interval = float(sample_interval)
        self.gpu_index = int(gpu_index)
        self.meter = GPUEnergyMeter(sample_interval=self.sample_interval, gpu_index=self.gpu_index)
        self.energy: Dict[str, float] = {}
        self.running = False
        self.start_time = 0.0
        self.idle_power_watts = 0.0
        self.idle_calibrated = False

    def calibrate_idle(self, duration_seconds: float = 2.0) -> float:
        """Calibrate baseline idle GPU power for dynamic-energy correction."""
        duration_seconds = max(0.0, float(duration_seconds))
        if duration_seconds <= 0.0:
            return self.idle_power_watts
        meter = GPUEnergyMeter(sample_interval=self.sample_interval, gpu_index=self.gpu_index)
        meter.start()
        time.sleep(duration_seconds)
        stats = meter.stop()
        self.idle_power_watts = float(stats.get("avg_power_watts", 0.0))
        self.idle_calibrated = self.idle_power_watts > 0.0
        logger.info(
            "Energy idle calibration complete: %.3f W (duration %.2fs)",
            self.idle_power_watts,
            duration_seconds,
        )
        return self.idle_power_watts

    def annotate_energy(self, stats: Dict[str, float], duration_seconds: float) -> Dict[str, float]:
        """Attach duration and dynamic-energy fields to a meter result."""
        out = dict(stats or {})
        duration_seconds = max(0.0, float(duration_seconds))
        total_joules = float(out.get("total_joules", 0.0))
        idle_energy = self.idle_power_watts * duration_seconds if self.idle_calibrated else 0.0
        dynamic_joules = max(0.0, total_joules - idle_energy)
        out["duration_seconds"] = duration_seconds
        out["idle_power_watts"] = float(self.idle_power_watts)
        out["idle_energy_joules"] = float(idle_energy)
        out["dynamic_joules"] = float(dynamic_joules)
        return out

    def start_span(self) -> Tuple[GPUEnergyMeter, float]:
        """Start an independent meter span for arbitrary code regions."""
        meter = GPUEnergyMeter(sample_interval=self.sample_interval, gpu_index=self.gpu_index)
        start_time = time.perf_counter()
        meter.start()
        return meter, start_time

    def stop_span(self, meter: Optional[GPUEnergyMeter], start_time: float) -> Dict[str, float]:
        """Stop an independent meter span and return normalized energy stats."""
        if meter is None:
            return self.annotate_energy({}, 0.0)
        stats = meter.stop()
        duration = time.perf_counter() - float(start_time)
        return self.annotate_energy(stats, duration)

    def on_rollout_start(self, agent: torch.nn.Module):
        """Called before rollout collection begins."""
        if not self.running:
            self.meter.start()
            self.running = True
            self.start_time = time.perf_counter()
            
            # Reset SNN counters if applicable
            if hasattr(agent, "reset_stats"):
                agent.reset_stats()

    def on_rollout_end(self, agent: torch.nn.Module, reward: float):
        """Called after rollout collection ends."""
        if self.running:
            raw = self.meter.stop()
            self.running = False
            self.energy = self.annotate_energy(raw, time.perf_counter() - self.start_time)
