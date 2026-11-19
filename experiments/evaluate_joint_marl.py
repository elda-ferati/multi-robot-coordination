
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from test_joint_marl import load_agent, evaluate


def main():
    torch.set_num_threads(2)
    agent = load_agent()

    policies = [
        "greedy",
        "sequential",
        "joint"
    ]

    results = {}

    print("\nJOINT MARL NEW-SCENARIO EVALUATION")
    print("=" * 65)

    for policy in policies:
        rows = [
            evaluate(seed, policy, agent)
            for seed in range(500, 530)
        ]

        results[policy] = {
            "success": sum(
                int(r["success"]) for r in rows
            ),
            "steps": sum(
                r["time_step"] for r in rows
            ) / len(rows),
            "distance": sum(
                r["distance"] for r in rows
            ) / len(rows),
            "collisions": sum(
                r["collisions"] for r in rows
            )
        }

        r = results[policy]

        print(
            f"{policy:12} "
            f"success={r['success']}/30 "
            f"steps={r['steps']:.3f} "
            f"distance={r['distance']:.3f} "
            f"collisions={r['collisions']}"
        )

    sequential = results["sequential"]["distance"]
    joint = results["joint"]["distance"]
    greedy = results["greedy"]["distance"]

    print("\nDISTANCE COMPARISON")

    print(
        "Joint vs Sequential:",
        f"{(sequential - joint) / sequential * 100:+.2f}%"
    )

    print(
        "Joint vs Greedy:",
        f"{(greedy - joint) / greedy * 100:+.2f}%"
    )

    print("\nJOINT MARL EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
