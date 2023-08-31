from tools import *
import matplotlib.pyplot as plt

def main():
    image = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)

    threshold = max_entropy_threshold(image)
    thresholded_image = image > threshold

    plt.imshow(thresholded_image, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
