
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from marl_allocation_v4 import (
    allocate_tasks_marl_v4
)

from benchmark_v3_v4_stress import run_scenario


def main():
    torch.set_num_threads(2)

    print("\nMARL V4 INTEGRATION TEST")
    print("=" * 60)

    result = run_scenario(
        100,
        "MARL V4",
        allocate_tasks_marl_v4
    )

    for key, value in result.items():
        print(f"{key}: {value}")

    assert result["error"] is None, (
        f"Simulation error: {result['error']}"
    )

    assert result["completed"] == result["total_tasks"], (
        "Not all tasks were completed"
    )

    assert result["collisions"] == 0, (
        "Collision detected"
    )

    print("\nMARL V4 INTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()
