import random
import math
import matplotlib.pyplot as plt
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

def tsp_dp_open_path(cost):
    """
    Dynamic Programming method.
    Starts at point 0.
    Visits every point once.
    Does not return to point 0.
    """

    n = len(cost)

    dp = {}
    parent = {}

    # Initial states: start from point 0 and go directly to point j
    for j in range(1, n):
        dp[(frozenset([j]), j)] = cost[0][j]
        parent[(frozenset([j]), j)] = 0

    # Build larger visited sets
    for size in range(2, n):
        for S in combinations(range(1, n), size):
            S = frozenset(S)

            for j in S:
                prev_set = S - {j}

                best_cost = float("inf")
                best_prev = None

                for k in prev_set:
                    new_cost = dp[(prev_set, k)] + cost[k][j]

                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_prev = k

                dp[(S, j)] = best_cost
                parent[(S, j)] = best_prev

    # All points except the starting point
    all_points = frozenset(range(1, n))

    # Choose the best ending point
    best_total_cost = float("inf")
    best_last = None

    for j in all_points:
        total_cost = dp[(all_points, j)]

        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_last = j

    # Reconstruct path
    path = [0]
    S = all_points
    last = best_last

    reverse_path = []

    while last != 0:
        reverse_path.append(last)
        prev = parent[(S, last)]
        S = S - {last}
        last = prev

    path += reverse_path[::-1]

    return best_total_cost, path


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
# Draw one route on one subplot
# --------------------------------------------------

def draw_route(ax, points, path, cost, title):
    """
    Draw one route using arrows and show edge costs.
    """

    # Plot points and point labels
    for i, (x, y) in enumerate(points):
        ax.scatter(x, y)
        ax.text(x + 0.1, y + 0.1, str(i), fontsize=12)

    # Draw arrows and edge costs
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

        # Middle point of the edge
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        edge_cost = cost[start_point][end_point]

        # Show edge cost
        ax.text(
            mid_x,
            mid_y,
            f"{edge_cost:.2f}",
            fontsize=9
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)


# --------------------------------------------------
# Main Program
# --------------------------------------------------

num_points = 10

# Generate the random points only once
points = generate_random_points(num_points)

# Generate one cost matrix using Euclidean distance
cost = euclidean_cost_matrix(points)

# Run Dynamic Programming method
dp_cost, dp_path = tsp_dp_open_path(cost)

# Run Nearest Neighbor method
nn_cost, nn_path = tsp_nearest_neighbor_open_path(cost, start=0)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("Random Points:")
for i, p in enumerate(points):
    print(i, p)

print("\nDynamic Programming")
print("Cost:", round(dp_cost, 2))
print("Path:", dp_path)

print("\nNearest Neighbor")
print("Cost:", round(nn_cost, 2))
print("Path:", nn_path)

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
    cost,
    f"Dynamic Programming Open Path\nTotal Cost = {dp_cost:.2f}"
)

draw_route(
    axes[1],
    points,
    nn_path,
    cost,
    f"Nearest Neighbor Open Path\nTotal Cost = {nn_cost:.2f}"
)

plt.tight_layout()
plt.show()