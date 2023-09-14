import numpy as np
import pandas as pd
from tools import *
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

arr = gray(cv2.imread('1.png'))

plt.imshow(arr)
plt.show()


sk = skeletonize(arr,  method= 'zhang')
plt.imshow(sk)
plt.show()

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

# Use the image coordinates for the position
pos = {i: (i % width, arr.shape[0] - i // width - 1) for i in G.nodes()}

# Draw the graph
nx.draw(G, pos, with_labels=True)
plt.gca().invert_yaxis()  # Invert y-axis to match the numpy array's coordinate system
plt.show()
