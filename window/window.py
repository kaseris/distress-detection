import cv2


class Window:

    def __init__(self, app_name):
        self.window_name = app_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def ShowFrame(self, img):
        cv2.imshow(self.window_name, img)

    def DestroyWindow(self):
        cv2.destroyWindow(self.window_name)

    @staticmethod
    def KillAllWindows():
        cv2.destroyAllWindows()
