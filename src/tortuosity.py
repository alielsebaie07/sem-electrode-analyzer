import heapq
import numpy as np


def calculate_tortuosity(binary_image):
    rows, cols = binary_image.shape

    start_points = [(0, c) for c in range(cols) if binary_image[0, c] == 0]
    end_points = set((rows - 1, c)
                     for c in range(cols) if binary_image[rows - 1, c] == 0)

    if not start_points or not end_points:
        return None, None

    start_points = start_points[::50]

    shortest_path = float('inf')
    best_path_coords = None

    for start in start_points:
        queue = [(0, start[0], start[1], [(start[0], start[1])])]
        visited = set()

        while queue:
            cost, r, c, path = heapq.heappop(queue)

            if (r, c) in visited:
                continue
            visited.add((r, c))

            if (r, c) in end_points:
                if cost < shortest_path:
                    shortest_path = cost
                    best_path_coords = path
                break

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and
                    0 <= nc < cols and
                    binary_image[nr, nc] == 0 and
                        (nr, nc) not in visited):
                    heapq.heappush(
                        queue, (cost + 1, nr, nc, path + [(nr, nc)]))

    if shortest_path == float('inf'):
        return None, None

    straight_line = rows
    tortuosity = shortest_path / straight_line
    return round(tortuosity, 3), best_path_coords


def draw_path_on_image(binary_image, path_coords):
    # Convert binary grayscale to RGB so we can draw a colored path
    rgb_image = np.stack([binary_image] * 3, axis=-1)

    if path_coords is None:
        return rgb_image

    # Draw path in bright cyan
    for r, c in path_coords:
        # Draw cyan path with thickness by coloring neighboring pixels too
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    rgb_image[nr, nc] = [0, 255, 255]

    return rgb_image
