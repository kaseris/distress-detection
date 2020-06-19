import cv2
import numpy as np

MODE_GRAYSCALE = 0

MODE_CENTER_CROP=0
MODE_NORMAL_CROP=1

class ImageProcessor:

    @staticmethod
    def ReadImage(path, mode=MODE_GRAYSCALE):
        img = cv2.imread(path, mode)
        return img

    @staticmethod
    def StoreImage(path, img):
        cv2.imwrite(path, img)

    @staticmethod
    def ConvertColorBGR2Gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def ResizeImage(img, dst_shape):
        return cv2.resize(img, dst_shape)

    @staticmethod
    def FFT(img):
        f = np.fft.fft2(img)
        return np.fft.fftshift(f)

    @staticmethod
    def IFFT(f):
        f = np.fft.ifftshift(f)
        return np.fft.ifft2(f)

    @staticmethod
    def LogTransform(f):
        return 20 * np.log(np.abs(f))

    @staticmethod
    def Crop(img, amount, mode=MODE_CENTER_CROP, *args):

        if mode == MODE_CENTER_CROP:
            # Find the center of the image.
            cx = img.shape[1] // 2
            cy = img.shape[0] // 2

            # Find the top left corner of the ROI.
            top_left = cy - amount // 2, cx - amount // 2

            return img[top_left[0]:top_left[0] + amount, top_left[1]:top_left[1]+amount]

        elif mode == MODE_NORMAL_CROP:
            if len(args) == 0:
                raise ValueError('The top-left corner of the ROI must be specified.')
            else:
                for arg in args:
                    if not (isinstance(arg, tuple) or isinstance(arg, list)):
                        raise TypeError("Argument must be either a tuple or a list.")
                    else:
                        return img[arg[0]:arg[0] + amount, arg[1]:arg[1]+amount]






