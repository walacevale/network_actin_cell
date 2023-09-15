import numpy as np
import pandas as pd
from tools import *
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

img_th = th_otsu(gray(cv2.imread('1.png')))


arr = skeletonize(img_th,  method= 'zhang')

def get_neighbors(i, j, arr):
    rows, cols = arr.shape

    # Vizinhos imediatos
    offsets_1 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    # Segundos vizinhos
    offsets_2 = [(-2, 0), (0, -2), (0, 2), (2, 0), (-2, -2), (-2, 2), (2, -2), (2, 2)]
    
    # Juntando todos os offsets
    all_offsets = offsets_1 + offsets_2

    neighbors = [(i + offset[0], j + offset[1]) for offset in all_offsets if 0 <= i + offset[0] < rows and 0 <= j + offset[1] < cols]
    
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

# Node specifications
node_size = 6
node_color = 'cyan'
node_edge_color = 'black'
node_alpha = 0.7

# Edge specifications
edge_width = 1.5
edge_alpha = 0.7
edge_color = 'gray'

# Node label specifications
label_font_size = 10
label_font_color = 'black'

# Draw the graph with the above specifications
nx.draw(G, pos,
        node_size=node_size, 
        node_color=node_color, 
        alpha=node_alpha, 
        edge_color=edge_color, 
        width=edge_width, 
        with_labels=False, 
        font_size=label_font_size,
        node_shape='o',
        edgecolors=node_edge_color)

# Invert y-axis to match the numpy array's coordinate system
#plt.gca().invert_yaxis()  

# Set a title (optional)
plt.title('Visualized Graph from Image')

# Set the graph background color
plt.gca().set_facecolor('lightgray')
plt.show()