import cv2


class FrameCapturer:

    def __init__(self):
        self.cap = cv2.VideoCapture(0) # Assume that we already have a webcam or a camera

    def CaptureFrame(self):
        _, frame = self.cap.read()
        return frame

    def Release(self):
        self.cap.release()