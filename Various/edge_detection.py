# https://datacarpentry.org/image-processing/08-edge-detection/
# import skimage
# import skimage.feature
# import skimage.viewer
import skimage
import cv2
import skimage.feature
import skimage.viewer
import skimage.io
import sys

# read command-line arguments
filename = "Template.jpg"
sigma = float(2)
low_threshold = float(0.1)
high_threshold = float(0.3)

# load and display original image as grayscale
image = skimage.io.imread(fname=filename, as_gray=True)
# le seguenti due righe non funzionano, non so perchè: le ho sostituite con imshow
# viewerr = skimage.viewer(image=image)
# viewerr.show()
cv2.imshow('Edges', image)

edges = skimage.feature.canny(
    image=image,
    sigma=sigma,
    low_threshold=low_threshold,
    high_threshold=high_threshold,
)

# display edges
viewerr = skimage.viewer.ImageViewer(edges)
viewerr.show()
