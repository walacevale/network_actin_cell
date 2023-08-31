import cv2
import numpy as np


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
