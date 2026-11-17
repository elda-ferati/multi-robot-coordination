
import heapq


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(environment, start, goal):
    if not environment.is_valid_position(*start):
        return None

    if not environment.is_valid_position(*goal):
        return None

    if environment.grid[start[0]][start[1]] != 0:
        return None

    if environment.grid[goal[0]][goal[1]] != 0:
        return None

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    while open_set:
        current = heapq.heappop(open_set)[1]

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

            tentative_g = g_score[current] + 1

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                f_score = tentative_g + heuristic(neighbor, goal)

                heapq.heappush(
                    open_set,
                    (f_score, neighbor)
                )

    return None
