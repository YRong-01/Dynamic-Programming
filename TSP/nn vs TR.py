import random
import math
import matplotlib.pyplot as plt
import time


# --------------------------------------------------
# Settings
# --------------------------------------------------

num_points = 100

# Change this if you want more lookahead steps.
# 1 means: try one next point, then finish the rest with nearest neighbor.
lookahead_steps = 2

random_seed = 3


# --------------------------------------------------
# Generate random points
# --------------------------------------------------

def generate_random_points(n, x_range=(0, 20), y_range=(0, 20), seed=1):
    random.seed(seed)

    points = []
    for _ in range(n):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        points.append((x, y))

    return points


# --------------------------------------------------
# Generate Euclidean cost matrix
# --------------------------------------------------

def euclidean_cost_matrix(points):
    n = len(points)
    cost = []

    for i in range(n):
        row = []
        for j in range(n):
            x1, y1 = points[i]
            x2, y2 = points[j]

            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            row.append(distance)

        cost.append(row)

    return cost


# --------------------------------------------------
# Compute cost of a path
# --------------------------------------------------

def path_cost(path, cost):
    total = 0

    for i in range(len(path) - 1):
        total += cost[path[i]][path[i + 1]]

    return total


# --------------------------------------------------
# Nearest Neighbor from current state
# --------------------------------------------------

def nearest_neighbor_from_state(cost, current, visited):
    n = len(cost)
    visited = set(visited)
    path = []
    total_cost = 0

    while len(visited) < n:
        nearest_point = None
        nearest_cost = float("inf")

        for j in range(n):
            if j not in visited and cost[current][j] < nearest_cost:
                nearest_cost = cost[current][j]
                nearest_point = j

        path.append(nearest_point)
        visited.add(nearest_point)
        total_cost += nearest_cost
        current = nearest_point

    return total_cost, path


# --------------------------------------------------
# Pure Nearest Neighbor Open Path
# --------------------------------------------------

def tsp_nearest_neighbor_open_path(cost, start=0):
    n = len(cost)

    visited = {start}
    path = [start]
    current = start
    total_cost = 0

    while len(visited) < n:
        nearest_point = None
        nearest_cost = float("inf")

        for j in range(n):
            if j not in visited and cost[current][j] < nearest_cost:
                nearest_cost = cost[current][j]
                nearest_point = j

        path.append(nearest_point)
        visited.add(nearest_point)
        total_cost += nearest_cost
        current = nearest_point

    return total_cost, path


# --------------------------------------------------
# Recursive lookahead rollout
# --------------------------------------------------

def rollout_value(cost, current, visited, depth):
    """
    Returns estimated future cost from current state.

    If depth == 0, finish the route using nearest neighbor.
    If depth > 0, try every possible next point and recursively look ahead.
    """

    n = len(cost)

    if len(visited) == n:
        return 0

    if depth == 0:
        future_cost, _ = nearest_neighbor_from_state(cost, current, visited)
        return future_cost

    best_cost = float("inf")

    for j in range(n):
        if j not in visited:
            new_visited = set(visited)
            new_visited.add(j)

            estimated_cost = cost[current][j] + rollout_value(
                cost,
                j,
                new_visited,
                depth - 1
            )

            if estimated_cost < best_cost:
                best_cost = estimated_cost

    return best_cost


# --------------------------------------------------
# Truncated Rollout Open Path
# --------------------------------------------------

def tsp_truncated_rollout_open_path(cost, start=0, lookahead_steps=1):
    """
    Truncated rollout.

    At each step:
    1. Try every unvisited point as the next point.
    2. Look ahead for a fixed number of steps.
    3. Finish the rest with nearest neighbor.
    4. Choose the next point with the lowest estimated total cost.
    """

    n = len(cost)

    visited = {start}
    path = [start]
    current = start
    total_cost = 0

    while len(visited) < n:
        best_next = None
        best_estimated_total = float("inf")

        for j in range(n):
            if j not in visited:
                new_visited = set(visited)
                new_visited.add(j)

                immediate_cost = cost[current][j]

                future_cost = rollout_value(
                    cost,
                    j,
                    new_visited,
                    lookahead_steps - 1
                )

                estimated_total = immediate_cost + future_cost

                if estimated_total < best_estimated_total:
                    best_estimated_total = estimated_total
                    best_next = j

        total_cost += cost[current][best_next]
        current = best_next
        visited.add(current)
        path.append(current)

    return total_cost, path


# --------------------------------------------------
# Draw route without point numbers
# --------------------------------------------------

def draw_route(ax, points, path, title):
    # Plot points only, no labels
    for x, y in points:
        ax.scatter(x, y)

    # Draw arrows
    for i in range(len(path) - 1):
        start_point = path[i]
        end_point = path[i + 1]

        x1, y1 = points[start_point]
        x2, y2 = points[end_point]

        dx = x2 - x1
        dy = y2 - y1

        ax.arrow(
            x1,
            y1,
            dx,
            dy,
            length_includes_head=True,
            head_width=0.15,
            head_length=0.25
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)


# --------------------------------------------------
# Main Program
# --------------------------------------------------

points = generate_random_points(num_points, seed=random_seed)
cost = euclidean_cost_matrix(points)


# Nearest Neighbor
start_time = time.time()
nn_cost, nn_path = tsp_nearest_neighbor_open_path(cost, start=0)
nn_time = time.time() - start_time


# Truncated Rollout
start_time = time.time()
rollout_cost, rollout_path = tsp_truncated_rollout_open_path(
    cost,
    start=0,
    lookahead_steps=lookahead_steps
)
rollout_time = time.time() - start_time


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nNearest Neighbor Open Path:")
print("Cost:", round(nn_cost, 2))
print("Time:", round(nn_time, 4), "seconds")

print("\nTruncated Rollout Open Path:")
print("Lookahead steps:", lookahead_steps)
print("Cost:", round(rollout_cost, 2))
print("Time:", round(rollout_time, 4), "seconds")

print("\nComparison:")
print("Nearest Neighbor Cost:", round(nn_cost, 2))
print("Truncated Rollout Cost:", round(rollout_cost, 2))

if rollout_cost < nn_cost:
    print("Truncated rollout found the shorter path.")
elif rollout_cost > nn_cost:
    print("Nearest neighbor found the shorter path.")
else:
    print("Both methods found the same cost.")


# --------------------------------------------------
# Plot results side by side
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

draw_route(
    axes[0],
    points,
    rollout_path,
    f"Truncated Rollout\nCost = {rollout_cost:.2f}, Time = {rollout_time:.4f}s"
)

draw_route(
    axes[1],
    points,
    nn_path,
    f"Nearest Neighbor\nCost = {nn_cost:.2f}, Time = {nn_time:.4f}s"
)



plt.tight_layout()
plt.show()