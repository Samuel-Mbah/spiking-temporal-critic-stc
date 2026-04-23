"""
Energy benchmarking suite for SNN vs. ANN comparison.

MEASUREMENT PHILOSOPHY
----------------------
Two distinct claims are tracked and must NOT be conflated in any paper:

  1. GPU Training/Inference Cost  (empirical, NVML wall-clock joules)
       → Valid for comparing *training cost* between ANN and SNN on identical
         hardware.  Does NOT demonstrate SNN efficiency: PyTorch executes spike
         tensors as dense CUDA ops regardless of sparsity.

  2. Theoretical Synaptic-Operation (SOP) Energy  (model-based estimate)
       → The standard NeurIPS / neuromorphic-RL claim.  Uses the AC vs MAC
         energy model (Horowitz 2014) to estimate what energy *would* be on
         neuromorphic hardware (Loihi, SpiNNaker, etc.).
         Formula:
           E_SNN = spike_count × n_synapses × E_AC   (AC  ≈ 0.9 pJ)
           E_ANN = activations × n_synapses × E_MAC  (MAC ≈ 4.6 pJ)

Suggested paper disclaimer:
  "We report theoretical energy estimates using the synaptic operation model
   (E_AC = 0.9 pJ, E_MAC = 4.6 pJ, following Horowitz 2014), as GPU hardware
   does not exploit spike sparsity.  Empirical GPU energy is reported
   separately for training-cost comparison."
"""

import time
import threading
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import logging

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
#  CONSTANTS — Horowitz (2014) 45 nm CMOS estimates
#  Cite as: M. Horowitz, "1.1 Computing's energy problem (and what we can
#  do about it)," ISSCC 2014.
# =======================================================================
E_AC_PJ:  float = 0.9    # pJ — accumulate only (SNN synaptic op)
E_MAC_PJ: float = 4.6    # pJ — multiply-accumulate (ANN neuron op)


# =======================================================================
#  HELPER: RECURSIVE STATS RETRIEVAL
# =======================================================================
def get_spike_stats_safe(module):
    if module is None:
        return {}
    if hasattr(module, "get_spike_stats"):
        return module.get_spike_stats()
    if hasattr(module, "actor"):
        return get_spike_stats_safe(module.actor)
    if hasattr(module, "backbone"):
        return get_spike_stats_safe(module.backbone)
    return {}


def reset_snn_stats(model):
    for m in model.modules():
        if hasattr(m, "reset_stats"):
            m.reset_stats()


def get_cumulative_spikes(model):
    total = 0.0
    total_elements = 0.0
    found = False
    for m in model.modules():
        if hasattr(m, "total_spikes") and hasattr(m, "total_timesteps"):
            ts = m.total_spikes
            tt = m.total_timesteps
            if isinstance(ts, torch.Tensor):
                ts = ts.detach().cpu().item()
            if isinstance(tt, torch.Tensor):
                tt = tt.detach().cpu().item()
            total += ts
            total_elements += tt
            found = True

    if not found:
        return 0.0, 0.0

    density  = total / total_elements if total_elements > 0 else 0.0
    sparsity = 1.0 - density
    return total, sparsity


def count_model_parameters(model: nn.Module) -> int:
    """Total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_synaptic_connections(model: nn.Module) -> int:
    """
    Approximate number of synaptic connections (fan-in × fan-out) by summing
    weight elements across Linear and Conv layers.  This is the standard
    denominator for SOP energy calculations.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if m.weight is not None:
                n += m.weight.numel()
    return n


def count_neurons(model: nn.Module) -> int:
    """Total output neurons across all Linear layers (neurons that receive synaptic input).

    Used to compute avg_fan_in = n_synapses / n_neurons for the SNN SOP formula.
    Each spike activates only the fan-in of its target neuron, not all synapses.
    """
    return sum(m.out_features for m in model.modules() if isinstance(m, nn.Linear))


# =======================================================================
#  PART 1: LOW-LEVEL GPU ENERGY METER
# =======================================================================
class GPUEnergyMeter:
    def __init__(self, sample_interval: float = 0.01, gpu_index: int = 0):
        self.sample_interval = sample_interval
        self._stop = False
        self.samples: List[tuple] = []
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
        power_readings: List[float] = []

        for i in range(1, len(self.samples)):
            t0, p0 = self.samples[i - 1]
            t1, _  = self.samples[i]
            if p0 is not None:
                delta_t = t1 - t0
                total_energy += p0 * delta_t
                power_readings.append(p0)
                valid_samples += 1

        avg_power  = float(np.mean(power_readings))  if power_readings else 0.0
        peak_power = float(np.max(power_readings))   if power_readings else 0.0
        return {
            "gpu_joules":     total_energy,
            "total_joules":   total_energy,
            "avg_power_watts": avg_power,
            "peak_power_watts": peak_power,
        }


# =======================================================================
#  PART 2: SOP THEORETICAL ENERGY MODEL
# =======================================================================
@dataclass
class SOPEnergyResult:
    """
    Theoretical energy estimate from the Synaptic Operation (SOP) model.
    These numbers represent *hypothetical neuromorphic hardware*, NOT GPU.
    """
    # --- inputs ---
    spike_count:       float = 0.0   # total spike events measured
    ann_activations:   float = 0.0   # total non-zero activations (ANN equivalent)
    n_synapses:        int   = 0     # synaptic connections in model

    # --- per-op constants used ---
    e_ac_pJ:  float = E_AC_PJ
    e_mac_pJ: float = E_MAC_PJ

    # --- results ---
    snn_theoretical_pJ:  float = 0.0
    ann_theoretical_pJ:  float = 0.0
    theoretical_speedup: float = 0.0   # ann / snn

    def summary(self) -> str:
        return (
            f"SOP Energy  →  SNN: {self.snn_theoretical_pJ/1e6:.4f} µJ  |  "
            f"ANN: {self.ann_theoretical_pJ/1e6:.4f} µJ  |  "
            f"Speedup: {self.theoretical_speedup:.2f}×  "
            f"(n_synapses={self.n_synapses:,}, "
            f"E_AC={self.e_ac_pJ} pJ, E_MAC={self.e_mac_pJ} pJ)"
        )


def compute_sop_energy(
    spike_count:     float,
    n_synapses:      int,
    ann_activations: float,
    e_ac_pJ:   float = E_AC_PJ,
    e_mac_pJ:  float = E_MAC_PJ,
    avg_fan_in: Optional[float] = None,
) -> SOPEnergyResult:
    """
    Theoretical energy estimate using the Synaptic Operation (SOP) model.

    This is the standard approach for claiming SNN energy efficiency in
    NeurIPS / RLC / ICLR papers when empirical neuromorphic hardware results
    are unavailable.  Must be clearly labelled as *theoretical* in any paper.

    Parameters
    ----------
    spike_count : float
        Total spike events fired across all SNN layers over the measured
        rollout.  Obtained from `get_cumulative_spikes()`.
    n_synapses : int
        Total synaptic connections (weight elements) in the model.
        Used as-is for the ANN energy baseline (each connection does one MAC per step).
    ann_activations : float
        Total inference steps for the ANN equivalent (= total_env_steps).
        ann_pJ = ann_activations × n_synapses × e_mac_pJ.
    e_ac_pJ : float
        Energy per accumulate operation in pJ.  Default 0.9 pJ (Horowitz 2014).
    e_mac_pJ : float
        Energy per multiply-accumulate operation in pJ.  Default 4.6 pJ.
    avg_fan_in : float, optional
        Average fan-in per neuron = n_synapses / n_neurons.
        Each spike activates avg_fan_in AC operations, not all n_synapses.
        Obtain via ``count_neurons(model)`` then ``n_synapses / n_neurons``.
        If None, falls back to n_synapses (old behaviour — overcounts SNN energy).

    Returns
    -------
    SOPEnergyResult
    """
    # Each SNN spike triggers avg_fan_in AC ops (one per incoming connection of
    # the receiving neuron), NOT n_synapses ops across the whole network.
    _fan_in = avg_fan_in if avg_fan_in is not None else n_synapses
    snn_pJ  = spike_count     * _fan_in    * e_ac_pJ
    ann_pJ  = ann_activations * n_synapses * e_mac_pJ
    speedup = (ann_pJ / snn_pJ) if snn_pJ > 0 else 0.0

    return SOPEnergyResult(
        spike_count=spike_count,
        ann_activations=ann_activations,
        n_synapses=n_synapses,
        e_ac_pJ=e_ac_pJ,
        e_mac_pJ=e_mac_pJ,
        snn_theoretical_pJ=snn_pJ,
        ann_theoretical_pJ=ann_pJ,
        theoretical_speedup=speedup,
    )


# =======================================================================
#  PART 3: BENCHMARKING METRICS DATACLASS
# =======================================================================
@dataclass
class EnergyMetrics:
    # ----- GPU wall-clock (empirical, hardware-specific) -----
    train_energy_joules:     float   # cumulative GPU J during training
    inference_energy_joules: float   # cumulative GPU J during inference benchmark
    total_energy_joules:     float   # train + inference
    dynamic_energy_joules:   float   # total minus idle baseline

    # ----- Per-episode / per-reward summaries -----
    energy_per_episode:  float
    energy_per_reward:   float
    energy_delay_product: float
    performance_per_watt: float
    throughput_per_watt:  float

    # ----- Per-step GPU joules (use raw_joules_per_env_step for
    #       apples-to-apples ANN vs SNN comparison on GPU) -----
    raw_joules_per_env_step:     Optional[float] = None   # total_J / env_steps  (same denom for both)
    dynamic_joules_per_env_step: Optional[float] = None   # dynamic_J / env_steps

    # NOTE: inference_joules_per_step retains legacy behaviour for existing
    # plots but should NOT be used for ANN/SNN comparison in papers because
    # the denominator differs (SNN divides by env_steps × T).
    # Use raw_joules_per_env_step instead.
    inference_joules_per_step: Optional[float] = None

    # ----- SNN spike statistics -----
    energy_per_spike:  Optional[float] = None   # GPU J / spike (informational only)
    sparsity_factor:   Optional[float] = None   # fraction of inactive neurons

    # ----- GPU power summary -----
    avg_power_watts:  float = 0.0
    idle_power_watts: float = 0.0
    peak_power_watts: float = 0.0

    # ----- Model structure (needed to reproduce SOP calculation) -----
    n_parameters:  int = 0
    n_synapses:    int = 0   # weight elements (fan-in × fan-out)

    # ----- Theoretical SOP energy (neuromorphic hardware estimate) -----
    # Populated by compute_sop_energy(); None if not an SNN or not computed.
    sop: Optional[SOPEnergyResult] = field(default=None, repr=False)


# =======================================================================
#  PART 4: HIGH-LEVEL BENCHMARKING FRAMEWORK
# =======================================================================
class EnergyBenchmark:
    def __init__(self):
        self.measurements: List[Dict[str, Any]] = []
        self.idle_power_watts = 0.0
        self.is_calibrated = False

    # ------------------------------------------------------------------
    def calibrate_idle(self, duration: float = 2.0):
        if not _NVML:
            return
        print(f"⚡ Calibrating idle power for {duration} seconds...")
        meter = GPUEnergyMeter()
        meter.start()
        time.sleep(duration)
        data = meter.stop()
        self.idle_power_watts = data.get("avg_power_watts", 0.0)
        self.is_calibrated = True
        print(f"⚡ Baseline Idle Power: {self.idle_power_watts:.3f} W")

    # ------------------------------------------------------------------
    def _compute_dynamic_energy(
        self,
        total_joules: float,
        episode_time: float,
        *,
        floor_fraction: float = 0.05,
    ) -> float:
        """
        Subtract idle baseline from measured energy.

        The naive `total - idle` can go negative for very short episodes with
        noisy NVML readings.  We clamp to a minimum of `floor_fraction` of the
        raw total so that downstream ratios remain well-behaved, while making
        the floor visible in the report.
        """
        idle_energy = self.idle_power_watts * episode_time
        raw_dynamic = total_joules - idle_energy
        floor_value = total_joules * floor_fraction
        return max(raw_dynamic, floor_value)

    # ------------------------------------------------------------------
    def measure_episode(
        self,
        model: torch.nn.Module,
        episode_fn: Callable[[torch.nn.Module], tuple],
        count_spikes: bool = True,
        warmup_runs:  int = 1,
        active_repeat: int = 1,
    ) -> Dict[str, Any]:

        warmup_runs   = max(0, int(warmup_runs))
        active_repeat = max(1, int(active_repeat))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for _ in range(warmup_runs):
            episode_fn(model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if count_spikes:
            reset_snn_stats(model)

        meter = GPUEnergyMeter(sample_interval=0.01)
        start_time = time.perf_counter()
        meter.start()

        reward = 0.0
        steps  = 0
        info   = {}
        for _ in range(active_repeat):
            r_i, s_i, info_i = episode_fn(model)
            reward += float(r_i)
            steps  += int(s_i)
            info    = info_i

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        data = meter.stop()

        episode_time   = end_time - start_time
        total_energy   = data["total_joules"]
        dynamic_energy = self._compute_dynamic_energy(total_energy, episode_time)

        spike_count = 0.0
        sparsity    = None
        if count_spikes:
            spike_count, sparsity = get_cumulative_spikes(model)

        return {
            "reward":             reward,
            "steps":              steps,
            "episodes_measured":  active_repeat,
            "time_seconds":       episode_time,
            "energy_joules":      total_energy,
            "dynamic_energy_joules": dynamic_energy,
            "power_watts":        total_energy / episode_time if episode_time > 0 else 0.0,
            "spike_count":        spike_count,
            "sparsity":           sparsity,
            "info":               info,
        }

    # ------------------------------------------------------------------
    def benchmark_model(
        self,
        model:             torch.nn.Module,
        episode_fn:        Callable[[torch.nn.Module], tuple],
        num_episodes:      int   = 100,
        model_type:        str   = "SNN",
        success_threshold: float = 475.0,
        prev_train_energy: float = 0.0,
        warmup_runs:       int   = 1,
        active_repeat:     int   = 1,
        # SOP parameters — only used when model_type == "SNN"
        sop_e_ac_pJ:  float = E_AC_PJ,
        sop_e_mac_pJ: float = E_MAC_PJ,
    ) -> EnergyMetrics:
        """
        Run the full benchmark and return an EnergyMetrics object.

        SOP theoretical energy is computed automatically for SNNs using
        spike counts collected during the benchmark.  Pass `sop_e_ac_pJ`
        and `sop_e_mac_pJ` to override the default Horowitz (2014) constants.
        """
        if not self.is_calibrated and _NVML:
            self.calibrate_idle()

        n_params   = count_model_parameters(model)
        n_synapses = estimate_synaptic_connections(model)

        measurements: List[Dict[str, Any]] = []
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

            if (i + 1) % max(1, num_episodes // 5) == 0:
                print(f"  ... {i + 1}/{num_episodes}")

        # --- aggregate ---
        inference_energy   = sum(m["energy_joules"]         for m in measurements)
        total_dynamic      = sum(m["dynamic_energy_joules"] for m in measurements)
        total_time         = sum(m["time_seconds"]          for m in measurements)
        total_reward       = sum(m["reward"]                for m in measurements)
        total_env_steps    = sum(m["steps"]                 for m in measurements)
        total_measured_eps = sum(m.get("episodes_measured", 1) for m in measurements)

        avg_power = inference_energy / total_time if total_time > 0 else 0.0

        # --- per-step GPU joules (identical denominator for both ANN and SNN) ---
        raw_joules_per_env_step     = None
        dynamic_joules_per_env_step = None
        if total_env_steps > 0:
            raw_joules_per_env_step     = inference_energy / total_env_steps
            dynamic_joules_per_env_step = total_dynamic    / total_env_steps

        # --- legacy inference_joules_per_step (kept for existing plots only) ---
        # For SNN this divides by env_steps × T; for ANN by env_steps.
        # Do NOT use this field for ANN/SNN comparisons in papers.
        inference_joules_per_step = None
        if model_type == "SNN":
            T: int = 1
            if   hasattr(model, "actor") and hasattr(model.actor, "T"): T = int(model.actor.T)
            elif hasattr(model, "T"):                                     T = int(model.T)
            total_inf_steps = total_env_steps * T
            if total_inf_steps > 0:
                inference_joules_per_step = float(total_dynamic / total_inf_steps)
        else:
            if total_env_steps > 0:
                inference_joules_per_step = float(total_dynamic / total_env_steps)

        # --- SNN-specific: spike stats + SOP theoretical energy ---
        energy_per_spike = None
        avg_sparsity     = None
        sop_result       = None

        if model_type == "SNN":
            total_spikes = sum(m["spike_count"] for m in measurements)

            if total_spikes > 0:
                energy_per_spike = inference_energy / total_spikes

            valid_sparsities = [m["sparsity"] for m in measurements if m["sparsity"] is not None]
            if valid_sparsities:
                avg_sparsity = float(np.mean(valid_sparsities))

            if total_spikes > 0 and n_synapses > 0:
                # Each spike triggers avg_fan_in AC ops, not all n_synapses.
                # avg_fan_in = n_synapses / n_neurons (total weights / receiving neurons).
                _n_neurons  = count_neurons(model)
                _avg_fan_in = n_synapses / _n_neurons if _n_neurons > 0 else float(n_synapses)
                sop_result = compute_sop_energy(
                    spike_count=total_spikes,
                    n_synapses=n_synapses,
                    ann_activations=float(total_env_steps),
                    e_ac_pJ=sop_e_ac_pJ,
                    e_mac_pJ=sop_e_mac_pJ,
                    avg_fan_in=_avg_fan_in,
                )

        else:
            # ANN SOP baseline: every weight connection does one MAC per env step.
            # spike_count=0 → snn_theoretical_pJ=0; ann_theoretical_pJ is the
            # reference energy used as the denominator in the SNN speedup claim.
            if n_synapses > 0 and total_env_steps > 0:
                _n_neurons  = count_neurons(model)
                _avg_fan_in = n_synapses / _n_neurons if _n_neurons > 0 else float(n_synapses)
                sop_result = compute_sop_energy(
                    spike_count=0.0,
                    n_synapses=n_synapses,
                    ann_activations=float(total_env_steps),
                    e_ac_pJ=sop_e_ac_pJ,
                    e_mac_pJ=sop_e_mac_pJ,
                    avg_fan_in=_avg_fan_in,
                )

        return EnergyMetrics(
            train_energy_joules=prev_train_energy,
            inference_energy_joules=inference_energy,
            total_energy_joules=prev_train_energy + inference_energy,
            dynamic_energy_joules=total_dynamic,
            energy_per_episode=inference_energy / max(1, total_measured_eps),
            energy_per_reward=inference_energy / total_reward if total_reward > 0 else float("inf"),
            inference_joules_per_step=inference_joules_per_step,
            raw_joules_per_env_step=raw_joules_per_env_step,
            dynamic_joules_per_env_step=dynamic_joules_per_env_step,
            energy_delay_product=inference_energy * total_time,
            performance_per_watt=total_reward / avg_power if avg_power > 0 else 0.0,
            throughput_per_watt=(total_measured_eps / total_time) / avg_power if avg_power > 0 else 0.0,
            energy_per_spike=energy_per_spike,
            sparsity_factor=avg_sparsity,
            avg_power_watts=avg_power,
            idle_power_watts=self.idle_power_watts,
            peak_power_watts=max(m["power_watts"] for m in measurements) if measurements else 0.0,
            n_parameters=n_params,
            n_synapses=n_synapses,
            sop=sop_result,
        )

    # ------------------------------------------------------------------
    def generate_report(
        self,
        snn_metrics: EnergyMetrics,
        ann_metrics: EnergyMetrics,
    ) -> str:
        def pct(a, b):
            return ((a - b) / a * 100) if a != 0 else 0.0

        def x_factor(a, b):
            return (a / b) if b != 0 else 0.0

        # Helper so optional floats print cleanly
        def fmt(val, fmt_str=":.6f", fallback="N/A"):
            return format(val, fmt_str.lstrip(":")) if val is not None else fallback

        # ---- GPU per-step: use raw_joules_per_env_step (same denominator) ----
        ann_gpu_per_step = ann_metrics.raw_joules_per_env_step
        snn_gpu_per_step = snn_metrics.raw_joules_per_env_step

        # ---- SOP section ----
        if snn_metrics.sop is not None:
            sop = snn_metrics.sop
            sop_section = f"""
3. THEORETICAL SOP ENERGY  ⚠️  neuromorphic hardware only, NOT GPU
--------------------------------------------------------------------
  (Horowitz 2014: E_AC = {sop.e_ac_pJ} pJ, E_MAC = {sop.e_mac_pJ} pJ)
  Synaptic connections (n_synapses): {sop.n_synapses:,}

  SNN theoretical energy : {sop.snn_theoretical_pJ/1e6:.4f} µJ
  ANN theoretical energy : {sop.ann_theoretical_pJ/1e6:.4f} µJ
  Theoretical speedup    : {sop.theoretical_speedup:.2f}×

  Spike count (measured) : {sop.spike_count:,.0f}
  Sparsity               : {f"{snn_metrics.sparsity_factor:.2%}" if snn_metrics.sparsity_factor else "N/A"}
  GPU J / spike          : {f"{(snn_metrics.energy_per_spike*1e9):.2f} nJ" if snn_metrics.energy_per_spike else "N/A"}

  ⚠️  Suggested paper wording:
      "We report theoretical energy estimates using the synaptic operation
       model (E_AC = {sop.e_ac_pJ} pJ, E_MAC = {sop.e_mac_pJ} pJ, Horowitz 2014), as GPU
       hardware does not exploit spike sparsity. Empirical GPU energy is
       reported separately for training-cost comparison."
"""
        else:
            sop_section = "\n3. THEORETICAL SOP ENERGY\n  N/A (ANN or spike stats not collected)\n"

        report = f"""
=== ENERGY BENCHMARK REPORT ===
(Idle Power Calibrated : {snn_metrics.idle_power_watts:.2f} W)
(ANN n_params          : {ann_metrics.n_parameters:,}  |  n_synapses: {ann_metrics.n_synapses:,})
(SNN n_params          : {snn_metrics.n_parameters:,}  |  n_synapses: {snn_metrics.n_synapses:,})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION A — GPU WALL-CLOCK ENERGY  (hardware-specific)
 NOTE: SNNs run as dense tensors on GPU; sparsity is NOT
 exploited here.  Use this section for training cost only.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. TOTAL CONSUMPTION
--------------------
  Total GPU Energy  (ANN) : {ann_metrics.total_energy_joules:.3f} J
  Total GPU Energy  (SNN) : {snn_metrics.total_energy_joules:.3f} J
  > Reduction              : {pct(ann_metrics.total_energy_joules, snn_metrics.total_energy_joules):.1f}%

  Dynamic GPU Energy (ANN) : {ann_metrics.dynamic_energy_joules:.3f} J
  Dynamic GPU Energy (SNN) : {snn_metrics.dynamic_energy_joules:.3f} J

2. PER-ENV-STEP GPU EFFICIENCY  (identical denominator for both)
----------------------------------------------------------------
  J / env-step (ANN) : {fmt(ann_gpu_per_step, ':.6f')}
  J / env-step (SNN) : {fmt(snn_gpu_per_step, ':.6f')}
  > GPU step speedup  : {f"{x_factor(ann_gpu_per_step, snn_gpu_per_step):.2f}×" if ann_gpu_per_step and snn_gpu_per_step else "N/A"}

  ⚠️  inference_joules_per_step (legacy field) uses DIFFERENT
      denominators for ANN (÷ env_steps) vs SNN (÷ env_steps×T).
      Do not use it for ANN/SNN comparisons — use J/env-step above.

  Avg GPU Power  (ANN) : {ann_metrics.avg_power_watts:.3f} W
  Avg GPU Power  (SNN) : {snn_metrics.avg_power_watts:.3f} W
  Peak GPU Power (ANN) : {ann_metrics.peak_power_watts:.3f} W
  Peak GPU Power (SNN) : {snn_metrics.peak_power_watts:.3f} W

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION B — THEORETICAL SOP ENERGY  (neuromorphic hardware)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{sop_section}"""

        print(report)
        return report


# =======================================================================
#  STANDALONE HELPER — compute SOP from saved benchmark outputs
# =======================================================================
def sop_from_saved_metrics(
    spike_count:     float,
    n_synapses:      int,
    total_env_steps: int,
    e_ac_pJ:  float = E_AC_PJ,
    e_mac_pJ: float = E_MAC_PJ,
) -> SOPEnergyResult:
    """
    Recompute SOP theoretical energy from previously saved benchmark data
    (e.g. loaded from benchmark_metrics.json).  Useful for post-hoc analysis
    without re-running the full benchmark.
    """
    ann_activations = float(total_env_steps)
    return compute_sop_energy(
        spike_count=spike_count,
        n_synapses=n_synapses,
        ann_activations=ann_activations,
        e_ac_pJ=e_ac_pJ,
        e_mac_pJ=e_mac_pJ,
    )


if __name__ == "__main__":
    print("Energy Benchmark Module Loaded.")