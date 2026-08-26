"""Render fixed-reference and episodic panels from a direct-delta checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anima_style_data.config import load_config
from anima_style_data.kv_activation_generator import sample_direct_reference_kv_delta_320


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    sample = config["kv_reference_direct_delta_320_sample"]
    sample["checkpoint"] = args.checkpoint
    sample["output_directory"] = args.output_directory
    result = sample_direct_reference_kv_delta_320(config, args.destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
