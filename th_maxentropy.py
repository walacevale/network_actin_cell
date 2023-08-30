import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, filters
import cv2 
import warnings
warnings.filterwarnings('ignore')

img = cv2.imread('1.png',  cv2.IMREAD_GRAYSCALE)

# Helper function for calculating entropy
def entp(x):
    temp = np.multiply(x, np.log(x))
    temp[np.isnan(temp)] = 0
    return temp

# Maximum entropy
H = cv2.calcHist([img],[0],None,[256],[0,256])
H /= H.sum()
theta = np.zeros(256)
Hf = np.zeros(256)
Hb = np.zeros(256)

for T in range(0,256):
    Hf[T] = - np.sum( entp(H[:T] /  np.sum(H[:T])) )

    Hb[T] = - np.sum( entp(H[T+1: ] / (1 - np.sum(H[:T])) ))
    theta[T] = Hf[T] + Hb[T]

theta_max = np.argmax(theta)

theta_max = np.argmax(theta)
img_out = img > theta_max

plt.imshow(img_out)
plt.show()




