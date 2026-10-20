import random
import math
import matplotlib.pyplot as plt
import time
from itertools import combinations


# --------------------------------------------------
# Generate random points
# --------------------------------------------------

def generate_random_points(n, x_range=(0, 10), y_range=(0, 10)):
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
# Dynamic Programming Open Path
# --------------------------------------------------

def tsp_dp_open_path_bitmask(cost, start=0):
    n = len(cost)

    # dp[mask][j] = minimum cost to start at "start",
    # visit the set of points in mask, and end at j
    dp = [[float("inf")] * n for _ in range(1 << n)]

    # parent[mask][j] = previous point before j
    parent = [[-1] * n for _ in range(1 << n)]

    # starting state: only start point visited
    start_mask = 1 << start
    dp[start_mask][start] = 0

    # loop through all visited sets
    for mask in range(1 << n):

        # skip states that do not include the start point
        if not (mask & start_mask):
            continue

        for j in range(n):

            # j must be inside the visited set
            if not (mask & (1 << j)):
                continue

            # if this state is unreachable, skip it
            if dp[mask][j] == float("inf"):
                continue

            # try going from j to each unvisited point k
            for k in range(n):
                if mask & (1 << k):
                    continue

                next_mask = mask | (1 << k)
                new_cost = dp[mask][j] + cost[j][k]

                if new_cost < dp[next_mask][k]:
                    dp[next_mask][k] = new_cost
                    parent[next_mask][k] = j

    # all points visited
    full_mask = (1 << n) - 1

    # open path: no need to return to start
    best_cost = float("inf")
    best_last = -1

    for j in range(n):
        if dp[full_mask][j] < best_cost:
            best_cost = dp[full_mask][j]
            best_last = j

    # reconstruct path
    path = []
    mask = full_mask
    current = best_last

    while current != -1:
        path.append(current)
        previous = parent[mask][current]
        mask = mask ^ (1 << current)
        current = previous

    path.reverse()

    return best_cost, path


# --------------------------------------------------
# Nearest Neighbor Open Path
# --------------------------------------------------

def tsp_nearest_neighbor_open_path(cost, start=0):
    """
    Nearest Neighbor method.
    Starts at point 0.
    Always goes to the closest unvisited point.
    Does not return to point 0.
    """

    n = len(cost)

    visited = [False] * n
    path = [start]
    visited[start] = True

    current = start
    total_cost = 0

    for _ in range(n - 1):
        nearest_point = None
        nearest_cost = float("inf")

        for j in range(n):
            if not visited[j] and cost[current][j] < nearest_cost:
                nearest_cost = cost[current][j]
                nearest_point = j

        path.append(nearest_point)
        visited[nearest_point] = True
        total_cost += nearest_cost
        current = nearest_point

    return total_cost, path


# --------------------------------------------------
# Draw one route without edge cost labels
# --------------------------------------------------

def draw_route(ax, points, path, title):
    """
    Draw one route using arrows.
    Edge costs are not shown on the plot.
    """

    # Plot points and point labels
    for i, (x, y) in enumerate(points):
        ax.scatter(x, y)
        ax.text(x + 0.1, y + 0.1, str(i), fontsize=12)

    # Draw arrows only
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
            head_width=0.2,
            head_length=0.3
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)


# --------------------------------------------------
# Main Program
# --------------------------------------------------

num_points = 20

points = generate_random_points(num_points)
cost = euclidean_cost_matrix(points)

dp_cost, dp_path = tsp_dp_open_path_bitmask(cost, start=0)
nn_cost, nn_path = tsp_nearest_neighbor_open_path(cost, start=0)

start_time = time.time()
dp_cost, dp_path = tsp_dp_open_path_bitmask(cost, start=0)
dp_time = time.time() - start_time

start_time = time.time()
nn_cost, nn_path = tsp_nearest_neighbor_open_path(cost, start=0)
nn_time = time.time() - start_time


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("Random Points:")
for i, p in enumerate(points):
    print(i, p)

print("\nDynamic Programming Open Path:")
print("Cost:", round(dp_cost, 2))
print("Path:", dp_path)
print("Time:", round(dp_time, 4), "seconds")

print("\nNearest Neighbor Open Path:")
print("Cost:", round(nn_cost, 2))
print("Path:", nn_path)
print("Time:", round(nn_time, 4), "seconds")

print("\nComparison:")
print("Dynamic Programming Cost:", round(dp_cost, 2))
print("Nearest Neighbor Cost:", round(nn_cost, 2))

if dp_cost < nn_cost:
    print("Dynamic Programming found the shorter path.")
elif dp_cost > nn_cost:
    print("Nearest Neighbor found the shorter path.")
else:
    print("Both methods found the same cost.")


# --------------------------------------------------
# Plot both results side by side
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

draw_route(
    axes[0],
    points,
    dp_path,
    f"Dynamic Programming Open Path\nTotal Cost = {dp_cost:.2f}"
)

draw_route(
    axes[1],
    points,
    nn_path,
    f"Nearest Neighbor Open Path\nTotal Cost = {nn_cost:.2f}"
)

plt.tight_layout()
plt.show()