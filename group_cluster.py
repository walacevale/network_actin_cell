# Import necessary libraries and tools
from tools import *
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import cv2

# Read the image in grayscale
image = cv2.imread('rect1.png', cv2.IMREAD_GRAYSCALE)

# Calculate the threshold value using max entropy threshold method and apply it to the image
threshold = max_entropy_threshold(image)
thresholded_image = image > threshold

# Label the connected components in the thresholded image
label_image = label(thresholded_image)

# Extract properties from the labeled image
regions = regionprops(label_image)
# Create a list of areas of the regions
areas = [region.area for region in regions]

# Loop over pairs of regions
for i in range(len(regions)):
    for j in range(i+1, len(regions)):
        # If the areas of two regions are roughly similar (within 10% difference)
        if 0.9*areas[j] <= areas[i] <= 1.1*areas[j]:
            # Assign the label of one region to the other, effectively merging them
            label_image[label_image == regions[j].label] = regions[i].label

# Overlay the labeled image on the original thresholded image for visualization
image_label_overlay = label2rgb(label_image, image=thresholded_image, bg_label=0)

# Plot the overlay image
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)
ax.set_axis_off()
plt.tight_layout()
plt.show()
