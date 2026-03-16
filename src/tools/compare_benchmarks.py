import json
import os
import argparse
import re

def parse_manual_report(filepath):
    """Parses the text report if JSON isn't available."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract Dynamic Energy
    dyn_match = re.search(r"SNN:\s+([\d\.]+)\s+J", content)
    dynamic_joules = float(dyn_match.group(1)) if dyn_match else 0.0
    
    return {"dynamic_energy_joules": dynamic_joules}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-dir", type=str, required=True, help="Path to ANN log directory")
    parser.add_argument("--snn-dir", type=str, required=True, help="Path to SNN log directory")
    args = parser.parse_args()

    # Load Metrics
    try:
        # Try loading JSON first (if you added JSON saving to SNN script)
        with open(os.path.join(args.ann_dir, "benchmark_metrics.json"), 'r') as f:
            ann_metrics = json.load(f)
    except FileNotFoundError:
        print("ANN benchmark JSON not found. Please run ann_baseline.py first.")
        return

    # For SNN, we parse the text report we just generated
    snn_report_path = os.path.join(args.snn_dir, "energy_report.txt")
    if os.path.exists(snn_report_path):
        snn_metrics = parse_manual_report(snn_report_path)
    else:
        print(f"SNN report not found at {snn_report_path}")
        return

    # Calculate Improvement
    ann_dyn = ann_metrics.get("dynamic_energy_joules", 0.0)
    snn_dyn = snn_metrics.get("dynamic_energy_joules", 0.0)

    if snn_dyn > 0:
        ratio = ann_dyn / snn_dyn
        reduction = (1 - snn_dyn / ann_dyn) * 100
    else:
        ratio = 0.0
        reduction = 0.0

    print("\n" + "="*40)
    print(f"⚡ FINAL HEAD-TO-HEAD COMPARISON ⚡")
    print("="*40)
    print(f"Dynamic Energy (ANN): {ann_dyn:.4f} J")
    print(f"Dynamic Energy (SNN): {snn_dyn:.4f} J")
    print("-" * 40)
    print(f"IMPROVEMENT RATIO:    {ratio:.2f}x")
    print(f"ENERGY REDUCTION:     {reduction:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
    
    
# python tools/compare_benchmarks.py --ann-dir results/logs/ann_baseline --snn-dir results/logs/snn_actor_direct