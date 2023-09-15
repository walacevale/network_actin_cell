import numpy as np
import pandas as pd
import cv2
import networkx as nx
from tools import *
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import glob
import os

# Configurações de diretórios
path_img_folder = "./Dados_image/temp/"
folder_original = glob.glob(path_img_folder + "*")


for img_path  in folder_original:

    img_th = th_otsu(gray(cv2.imread(img_path )))
    sk = skeletonize(img_th,  method= 'zhang').astype(int)*255

    dic,edges,nodes=nodEdg(sk)

    G=nx.from_pandas_edgelist(edges)
    fig, ax = plt.subplots(figsize = (40, 30))
    pos = dic  # position layout
    print(len(nodes))

    #nx.draw_networkx(G, pos, nodelist= nodes, node_size=2,with_labels=False, node_color="white",edgecolors='black', linewidths=5, alpha=1 , width = 5)
    #ax.set_axis_off()
    #fig.tight_layout()
    #plt.show()
