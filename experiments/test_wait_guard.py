
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from test_marl_episodes import create_episode
from evaluate_marl_v2 import load_v2
from marl_environment_v2 import MARLEnvironmentV2
from marl_wait_policy import select_actions_with_wait
from marl_wait_guard import WaitGuardPolicy

def evaluate(seed, model, guarded):
    original = create_episode(seed)

    env = MARLEnvironmentV2(
        original.sim,
        max_steps=original.max_steps,
        max_candidates=original.max_candidates
    )

    guard = WaitGuardPolicy(
        model, max_wait=2
    )

    waits = 0
    opportunities = 0

    for _ in range(env.max_steps):
        obs = env.get_observations()

        if guarded:
            actions = guard.select_actions(obs)
        else:
            actions = select_actions_with_wait(
                model, obs
            )

        for rid, candidates in obs.items():
            if candidates:
                opportunities += 1
                if actions[rid] is None:
                    waits += 1

        _, _, done, info = env.step(actions)

        if done:
            break

    return info, waits, opportunities

def main():
    torch.set_num_threads(2)
    model = load_v2()

    print("\nWAIT GUARD DIAGNOSTIC")
    print("=" * 60)

    for guarded in (False, True):
        results = [
            evaluate(seed, model, guarded)
            for seed in range(600, 630)
        ]

        infos = [r[0] for r in results]

        success = sum(
            int(r["success"]) for r in infos
        )

        steps = sum(
            r["time_step"] for r in infos
        ) / len(infos)

        distance = sum(
            r["distance"] for r in infos
        ) / len(infos)

        waits = sum(r[1] for r in results)
        opportunities = sum(
            r[2] for r in results
        )

        wait_rate = (
            100 * waits / opportunities
            if opportunities else 0
        )

        label = "GUARDED" if guarded else "ORIGINAL"

        print(
            f"{label:10} "
            f"success={success}/30 "
            f"steps={steps:.2f} "
            f"distance={distance:.2f} "
            f"wait={wait_rate:.2f}%"
        )

    print("\nWAIT GUARD DIAGNOSTIC COMPLETE")

if __name__ == "__main__":
    main()