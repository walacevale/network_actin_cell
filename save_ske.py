from tools import *
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import glob
import os

# Configurações de diretórios
path_img_folder = "./Dados_image/\\FIBRO CITO T0 COL/"
save_folder = './skeletonize/T0/'

folder_original = glob.glob(path_img_folder + "*")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for img_path  in folder_original:

    img_th = th_otsu(gray(cv2.imread(img_path )))
    sk = skeletonize(img_th,  method= 'zhang').astype(int)*255
    

    base_filename = os.path.basename(img_path)
    save_path = os.path.join(save_folder, base_filename)

    cv2.imwrite(save_path, sk)