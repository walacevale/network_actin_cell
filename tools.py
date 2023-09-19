import cv2
import numpy as np
from skimage.filters import threshold_otsu
import pandas as pd

def calculate_entropy(probabilities):
    """
    Calculate entropy for a given probability distribution.
    The value 'epsilon' is necessary for avoid divide by zero in log 0.
    """
    epsilon = 1e-23
    entropy_values = -np.multiply(probabilities, np.log(probabilities + epsilon))
    entropy_values[np.isnan(entropy_values)] = 0
    return np.sum(entropy_values)


def max_entropy_threshold(image):
    """
    Calculate the threshold value using maximum entropy method.
    """
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram /= histogram.sum()

    foreground_entropy = np.zeros(256)
    background_entropy = np.zeros(256)

    for threshold in range(256):
        foreground_prob = histogram[:threshold] / np.sum(histogram[:threshold])
        background_prob = histogram[threshold+1:] / (1 - np.sum(histogram[:threshold]))
        
        foreground_entropy[threshold] = calculate_entropy(foreground_prob)
        background_entropy[threshold] = calculate_entropy(background_prob)

    total_entropies = foreground_entropy + background_entropy
    optimal_threshold = np.argmax(total_entropies)
    
    return optimal_threshold

def th_otsu(image):
    """
    Apply Otsu's thresholding method to the image.
    
    Args:
    - image (numpy.ndarray): Input grayscale image.
    
    Returns:
    - numpy.ndarray: Binary image after Otsu's thresholding.
    """
    thresh_value = threshold_otsu(image)
    binary = image > thresh_value
    return (np.array(binary, dtype=int)) * 255

def gray(image):
    """
    Convert a BGR image to grayscale.
    
    Args:
    - image (numpy.ndarray): Input BGR image.
    
    Returns:
    - numpy.ndarray: Grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def viz (N,L):
    '''
    Return the neighbors of the site N for a lattice with horizontal shape L.
    '''
    k=[N-L,N+L,N+1,N-1]
    return k

def viz2 (N,L):
    '''
    Return the neighbors of the site N for a lattice with horizontal shape L.
    '''
    k=[N-L,N+L,N+1,N-1,N-L+1,N-L-1,N+L+1,N+L-1]
    return k

def nodEdg(Image):
    '''
    Return a dictionary with the lattice positions of the nodes, an data frame with source and target connection
    between the nodes and the nodes.
    '''
    L=Image.shape[1]
    edges=[]
    
    Npos=np.where(Image == 255)
    Nodes=Npos[0]*Image.shape[0]+Npos[1]
    Spos=np.flip(Npos,axis=0)
    Spos[1]= Image.shape[1] - Spos[1]
    dic= dict(zip(Nodes, zip(*Spos)))
    
    for i in range (len(Nodes)):
        nb=viz2(Nodes[i],L)           #viz-primeiros vizinhos  viz2-primeiros e segundos vizinhos
        for j in range (len(nb)):
            if nb[j] in Nodes:
                if (nb[j],Nodes[i]) not in edges:
                    edges.append((Nodes[i],nb[j]))
    df=pd.DataFrame()
    df['source'] = np.array(edges)[:,0]
    df['target'] = np.array(edges)[:,1]
    
    return dic,df,Nodes
