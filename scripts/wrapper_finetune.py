#!/usr/bin/env python3
"""Wrapper script to automate weekend fine-tuning of Kronos.
Runs fine-tuning for a specific set of symbols and updates the default.yaml
to point to the newly trained weights.
"""

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ["SPY", "QQQ"]
EPOCHS = 10


def main():
    print("Starting automated weekend fine-tuning for Kronos...")

    for symbol in SYMBOLS:
        print(f"\n--- Fine-tuning {symbol} ---")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "finetune_kronos.py"),
            "--symbol",
            symbol,
            "--epochs",
            str(EPOCHS),
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully fine-tuned {symbol}")
        except subprocess.CalledProcessError as e:
            print(f"Error fine-tuning {symbol}: {e}")
            continue

    # Update configs/default.yaml to use the SPY model path
    # (acts as a broad market proxy for the single-path config structure).
    print("\nUpdating configs/default.yaml with new weights path...")
    config_path = ROOT / "configs" / "default.yaml"

    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        if "kronos" not in cfg:
            cfg["kronos"] = {}

        model_dir = "data/models/kronos/SPY/model"
        cfg["kronos"]["finetuned_model_path"] = model_dir

        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        print(f"Config updated: kronos.finetuned_model_path = {model_dir}")

    except Exception as e:
        print(f"Failed to update default.yaml: {e}")


if __name__ == "__main__":
    main()
