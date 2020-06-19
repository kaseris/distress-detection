from __future__ import absolute_import

import matplotlib.pyplot as plt

from ..core.image_processor import ImageProcessor


def ShowImage(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

def ShowTransformedImage(f):
    f_demo = ImageProcessor.LogTransform(f)
    plt.figure()
    plt.imshow(f_demo, cmap='gray')
    plt.show()