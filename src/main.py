
from environment import GridEnvironment


# Create warehouse environment
environment = GridEnvironment(10, 10)

# Static obstacles (walls and shelves)
environment.add_obstacle(1, 1)
environment.add_obstacle(1, 2)
environment.add_obstacle(1, 3)

# Add 5 random static obstacles
environment.add_random_obstacles(5)

# Add 2 initial dynamic obstacles
environment.add_random_dynamic_obstacle()
environment.add_random_dynamic_obstacle()

print("INITIAL ENVIRONMENT:")
environment.display()
print("Dynamic obstacles:", len(environment.dynamic_obstacles))
print("Positions:", environment.dynamic_obstacles)

# Simulate 6 time steps
for step in range(1, 7):

    # Move dynamic obstacles
    environment.move_dynamic_obstacles()

    # Add a new obstacle at time step 2
    if step == 2:
        environment.add_random_dynamic_obstacle()

    # Remove an obstacle at time step 4
    if step == 4:
        environment.remove_random_dynamic_obstacle()

    print(f"\nTIME STEP {step}:")
    environment.display()

    print("Dynamic obstacles:", len(environment.dynamic_obstacles))
    print("Positions:", environment.dynamic_obstacles)
