
import random


class GridEnvironment:
    def __init__(self, rows, cols):
        if rows <= 0 or cols <= 0:
            raise ValueError("Grid dimensions must be positive.")

        self.rows = rows
        self.cols = cols

        self.grid = []
        self.dynamic_obstacles = []

        for i in range(rows):
            row = []

            for j in range(cols):
                row.append(0)

            self.grid.append(row)

    def is_valid_position(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def add_obstacle(self, row, col):
        if self.is_valid_position(row, col) and self.grid[row][col] == 0:
            self.grid[row][col] = 1
        else:
            print("Cannot place static obstacle here.")

    def add_random_obstacles(self, number_of_obstacles):
        available_cells = sum(row.count(0) for row in self.grid)

        if number_of_obstacles < 0 or number_of_obstacles > available_cells:
            raise ValueError("Invalid number of obstacles.")

        added = 0

        while added < number_of_obstacles:
            row = random.randint(0, self.rows - 1)
            col = random.randint(0, self.cols - 1)

            if self.grid[row][col] == 0:
                self.grid[row][col] = 1
                added += 1

    def add_dynamic_obstacle(self, row, col):
        if self.is_valid_position(row, col) and self.grid[row][col] == 0:
            self.grid[row][col] = 2
            self.dynamic_obstacles.append((row, col))
            return True

        print("Cannot place dynamic obstacle here.")
        return False

    def move_dynamic_obstacles(self):
        old_positions = self.dynamic_obstacles.copy()
        new_positions = []

        # Reserve all positions occupied at the start
        occupied_positions = set(old_positions)

        directions = [
            (-1, 0),  # Up
            (1, 0),   # Down
            (0, -1),  # Left
            (0, 1)    # Right
        ]

        for row, col in old_positions:
            valid_positions = []

            for dr, dc in directions:
                new_row = row + dr
                new_col = col + dc

                if self.is_valid_position(new_row, new_col):
                    if self.grid[new_row][new_col] == 0:
                        if (new_row, new_col) not in occupied_positions:
                            valid_positions.append((new_row, new_col))

            if valid_positions:
                new_row, new_col = random.choice(valid_positions)
            else:
                new_row, new_col = row, col

            new_positions.append((new_row, new_col))
            occupied_positions.add((new_row, new_col))

        # Clear old dynamic obstacle positions
        for row, col in old_positions:
            self.grid[row][col] = 0

        # Place dynamic obstacles at their new positions
        for row, col in new_positions:
            self.grid[row][col] = 2

        self.dynamic_obstacles = new_positions

    def add_random_dynamic_obstacle(self):
        free_positions = []

        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] == 0:
                    free_positions.append((row, col))

        if not free_positions:
            return False

        row, col = random.choice(free_positions)
        return self.add_dynamic_obstacle(row, col)

    def remove_random_dynamic_obstacle(self):
        if not self.dynamic_obstacles:
            return False

        row, col = random.choice(self.dynamic_obstacles)

        self.grid[row][col] = 0
        self.dynamic_obstacles.remove((row, col))

        return True

    def display(self):
        for row in self.grid:
            print(row)
