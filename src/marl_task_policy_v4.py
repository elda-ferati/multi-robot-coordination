
import torch


def select_task_actions(model, observations):
    """
    Joint task assignment without learned WAIT.

    Each task is assigned to at most one robot.
    Each robot receives at most one task.
    """
    actions = {
        rid: None for rid in observations
    }

    bids = []

    model.eval()

    with torch.no_grad():
        for rid, candidates in observations.items():
            if not candidates:
                continue

            features = torch.tensor(
                [c["features"] for c in candidates],
                dtype=torch.float32
            )

            scores = model(features).reshape(-1)

            for candidate, score in zip(
                candidates, scores
            ):
                bids.append((
                    float(score.item()),
                    rid,
                    candidate["task_id"]
                ))

    bids.sort(
        key=lambda item: (
            -item[0], item[1], item[2]
        )
    )

    assigned_robots = set()
    assigned_tasks = set()

    for score, rid, task_id in bids:
        if rid in assigned_robots:
            continue

        if task_id in assigned_tasks:
            continue

        actions[rid] = task_id
        assigned_robots.add(rid)
        assigned_tasks.add(task_id)

    return actions
