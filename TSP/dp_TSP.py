import random
import math
import matplotlib.pyplot as plt
from itertools import combinations


def generate_random_points(n, x_range=(0, 10), y_range=(0, 10)):
    """
    Generate n random 2D points.
    """
    points = []

    for _ in range(n):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        points.append((x, y))

    return points


def euclidean_cost_matrix(points):
    """
    Generate a cost matrix using Euclidean distance.
    cost[i][j] = distance from point i to point j
    """
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


def tsp_dp(cost):
    """
    Solve TSP using dynamic programming.
    Starts and ends at point 0.
    """
    n = len(cost)

    dp = {}
    parent = {}

    for j in range(1, n):
        dp[(frozenset([j]), j)] = cost[0][j]
        parent[(frozenset([j]), j)] = 0

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

    all_points = frozenset(range(1, n))

    best_total_cost = float("inf")
    best_last = None

    for j in all_points:
        total_cost = dp[(all_points, j)] + cost[j][0]

        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_last = j

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
    path.append(0)

    return best_total_cost, path


def plot_tsp(points, path):
    """
    Plot points and the TSP route.
    """
    x = [points[i][0] for i in path]
    y = [points[i][1] for i in path]

    plt.figure(figsize=(7, 6))

    plt.plot(x, y, marker="o")

    for i, (px, py) in enumerate(points):
        plt.text(px + 0.1, py + 0.1, str(i), fontsize=12)

    # Draw arrows
    for i in range(len(path) - 1):
        start_point = path[i]
        end_point = path[i + 1]

        x1, y1 = points[start_point]
        x2, y2 = points[end_point]

        dx = x2 - x1
        dy = y2 - y1

        plt.arrow(
            x1, y1,
            dx, dy,
            length_includes_head=True,
            head_width=0.2,
            head_length=0.3
        )
    plt.title("TSP Route with Random Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


# ---------------- Main Program ----------------

num_points = 20

points = generate_random_points(num_points)
cost = euclidean_cost_matrix(points)

best_cost, best_path = tsp_dp(cost)

print("Points:")
for i, p in enumerate(points):
    print(i, p)

print("\nBest cost:", best_cost)
print("Best path:", best_path)

plot_tsp(points, best_path)