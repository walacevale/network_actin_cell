import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


arr = np.zeros((10, 10), dtype=np.uint8)


arr[2:5, 2:5] = [[255,255, 255],
                 [255,255, 255],
                 [255,255, 255]]

arr[5:8, 5:8] = [[255,255, 255],
                 [255,255, 255],
                 [255,255, 255]]

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

G = nx.Graph(df)
pos = nx.spring_layout(G, seed=1)
nx.draw(G, pos)
plt.show()