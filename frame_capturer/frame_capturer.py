import cv2

from ..core.image_processor import ImageProcessor


class FrameCapturer:

    def __init__(self):
        self.cap = cv2.VideoCapture(0) # Assume that we already have a webcam or a camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

    def CaptureFrame(self):
        _, frame = self.cap.read()
        frame = ImageProcessor.ConvertColorBGR2Gray(frame)

        return frame

    def Release(self):
        self.cap.release()