import numpy as np
import pandas as pd

arr = np.zeros((10, 10), dtype=np.uint8)


arr[2:5, 2:5] = [[50, 100, 150],
                 [50, 100, 150],
                 [50, 100, 150]]

arr[5:8, 5:8] = [[255, 205, 155],
                 [255, 205, 155],
                 [255, 205, 155]]

print('arr : ', arr )

def get_neighbors(i, j, arr):
    rows, cols = arr.shape
    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    neighbors = [(i + offset[0], j + offset[1]) for offset in offsets if 0 <= i + offset[0] < rows and 0 <= j + offset[1] < cols]
    return neighbors

def get_pixel_label(i, j, width):
    return i * width + j

connections = []
width = arr.shape[1]

for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if arr[i, j] != 0:
            for ni, nj in get_neighbors(i, j, arr):
                if arr[ni, nj] != 0:
                    pixel_label = get_pixel_label(i, j, width)
                    neighbor_pixel_label = get_pixel_label(ni, nj, width)
                    intensity_diff = 1
                    connections.append([pixel_label, neighbor_pixel_label, intensity_diff])

df = pd.DataFrame(connections, columns=["source", "target", "weight"])

print(df)