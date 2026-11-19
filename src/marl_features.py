
def get_robot_task_features(
    robot, task, robots, tasks, environment
):
    """
    Return 12 normalized features describing
    one robot-task pair.
    """

    rows = len(environment.grid)
    cols = len(environment.grid[0])

    row_scale = max(1, rows - 1)
    col_scale = max(1, cols - 1)
    distance_scale = max(1, rows + cols - 2)

    robot_row, robot_col = robot.position
    pickup_row, pickup_col = task.pickup
    delivery_row, delivery_col = task.delivery

    pickup_distance = (
        abs(robot_row - pickup_row)
        + abs(robot_col - pickup_col)
    )

    delivery_distance = (
        abs(pickup_row - delivery_row)
        + abs(pickup_col - delivery_col)
    )

    busy_robots = sum(
        r.current_task is not None
        for r in robots
    )

    pending_tasks = sum(
        t.status == "pending"
        for t in tasks
    )

    blocked_cells = sum(
        cell != 0
        for row in environment.grid
        for cell in row
    )

    return [
        robot_row / row_scale,
        robot_col / col_scale,
        pickup_row / row_scale,
        pickup_col / col_scale,
        delivery_row / row_scale,
        delivery_col / col_scale,
        pickup_distance / distance_scale,
        delivery_distance / distance_scale,
        min(task.priority, 3) / 3,
        busy_robots / max(1, len(robots)),
        pending_tasks / max(1, len(tasks)),
        blocked_cells / (rows * cols),
    ]
