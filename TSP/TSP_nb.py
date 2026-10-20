import random
import math
import matplotlib.pyplot as plt


def generate_random(n, x_range=(0, 10), y_range=(0, 10)):

# generate the random points in a 2D space 

    points = []

    for _ in range(n):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        points.append((x, y))

    return points


def euclidean_cost(points):

# use Euclidean distance beween two points as the cost function and store it in the cose matrix

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


def tsp_nearest_neighbor(cost, start=0):

# take the point that is closest to the current point

    n = len(cost)

    visited = [False] * n # initially all points are false (not visited)
    path = [start] # start from point 0
    visited[start] = True # mark the point 0 as visited

    current = start # the position right now
    total_cost = 0

    for _ in range(n - 1):
        nearest_point = None # don't know the closest point yet
        nearest_cost = float("inf") # starts as infinity so that any real distance is smaller than it.

        for j in range(n): 
            if not visited[j] and cost[current][j] < nearest_cost: # if j is not visited yet and the distance from the current point to point j is smaller than the best distance found so far
                nearest_cost = cost[current][j]
                nearest_point = j

        path.append(nearest_point) # adds the selected point to the route
        visited[nearest_point] = True # mark the point as visited
        total_cost += nearest_cost
        current = nearest_point

    # if want to return to the starting point
    # total_cost += cost[current][start]
    # path.append(start)

    return total_cost, path


def plot_tsp(points, path):

    plt.figure(figsize=(8, 8))

    # Plot points
    for i, (x, y) in enumerate(points):
        plt.scatter(x, y)
        plt.text(x + 0.1, y + 0.1, str(i), fontsize=12)


    for i in range(len(path) - 1):
        start_point = path[i]
        end_point = path[i + 1]

        x1, y1 = points[start_point]
        x2, y2 = points[end_point]

        dx = x2 - x1
        dy = y2 - y1

        plt.arrow(x1, y1, dx, dy, length_includes_head=True, head_width=0.2, head_length=0.3)

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        path_cost = cost[start_point][end_point]

        plt.text(
            mid_x,
            mid_y,
            f"{path_cost:.2f}",
            fontsize=10
        )

    plt.title("TSP Route Using Nearest Neighbor")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


# Main routine

num_points = 150

points = generate_random(num_points)
cost = euclidean_cost(points)

best_cost, best_path = tsp_nearest_neighbor(cost, start=0)

print("Points:")
for i, p in enumerate(points):
    print(i, p)

print("\nNearest Neighbor Cost:", best_cost)
print("Nearest Neighbor Path:", best_path)

plot_tsp(points, best_path)