
import heapq


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(environment, start, goal, blocked_positions=None):
    blocked = set(blocked_positions or [])

    if not environment.is_valid_position(*start):
        return None

    if not environment.is_valid_position(*goal):
        return None

    if start in blocked or goal in blocked:
        return None

    if environment.grid[start[0]][start[1]] != 0:
        return None

    if environment.grid[goal[0]][goal[1]] != 0:
        return None

    open_set = [(heuristic(start, goal), start)]
    came_from = {}
    g_score = {start: 0}

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            return path[::-1]

        for dr, dc in directions:
            neighbor = (
                current[0] + dr,
                current[1] + dc
            )

            row, col = neighbor

            if not environment.is_valid_position(row, col):
                continue

            if environment.grid[row][col] != 0:
                continue

            if neighbor in blocked:
                continue

            new_cost = g_score[current] + 1

            if new_cost < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = new_cost
                came_from[neighbor] = current

                priority = new_cost + heuristic(neighbor, goal)

                heapq.heappush(
                    open_set,
                    (priority, neighbor)
                )

    return None
