
import torch

class WaitGuardPolicy:
    def __init__(self, model, max_wait=2):
        self.model = model
        self.max_wait = max_wait
        self.streaks = {}

    def select_actions(self, observations):
        actions = {
            rid: None for rid in observations
        }

        bids = []
        wait_scores = {}

        self.model.eval()

        with torch.no_grad():
            for rid, candidates in observations.items():
                if not candidates:
                    self.streaks[rid] = 0
                    continue

                x = torch.tensor(
                    [c["features"] for c in candidates],
                    dtype=torch.float32
                )

                scores = self.model(x)
                wait_scores[rid] = float(
                    self.model.score_wait(x).item()
                )

                for candidate, score in zip(
                    candidates, scores
                ):
                    bids.append((
                        float(score.item()),
                        rid,
                        candidate["task_id"]
                    ))

        assigned_robots = set()
        assigned_tasks = set()

        def assign(score, rid, tid):
            if rid in assigned_robots:
                return False
            if tid in assigned_tasks:
                return False

            actions[rid] = tid
            assigned_robots.add(rid)
            assigned_tasks.add(tid)
            return True

        # First handle robots that have waited
        # for too many consecutive opportunities.
        forced = [
            bid for bid in bids
            if self.streaks.get(bid[1], 0)
            >= self.max_wait
        ]

        forced.sort(key=lambda x: (-x[0], x[1], x[2]))

        for score, rid, tid in forced:
            assign(score, rid, tid)

        # Normal decisions: compare task vs WAIT.
        bids.sort(key=lambda x: (-x[0], x[1], x[2]))

        for score, rid, tid in bids:
            if score <= wait_scores[rid]:
                continue

            assign(score, rid, tid)

        for rid, candidates in observations.items():
            if not candidates:
                self.streaks[rid] = 0

            elif actions[rid] is None:
                self.streaks[rid] = (
                    self.streaks.get(rid, 0) + 1
                )

            else:
                self.streaks[rid] = 0

        return actions