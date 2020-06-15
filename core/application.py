from .callbacks import CallbackList
from .callbacks import TimeElapsedCallback, TerminateAppCallback

from ..frame_capturer import FrameCapturer
from ..window import Window



class Application:
    def __init__(self, app_name):
        self.appName = app_name
        self.isAppRunning = False

    def run(self):
        raise NotImplementedError("run() must be implemented from a subclass.")

    def terminateApplicationLoop(self):
        self.isAppRunning = False

class RemoteGatewayApplication(Application):

    def __init__(self, app_name, callbacks=None):
        super(RemoteGatewayApplication, self).__init__(app_name)

        self.callbacks = callbacks or []
        self.callbacks.append(TimeElapsedCallback())
        self.callbacks.append(TerminateAppCallback(self))
        self.cblist = CallbackList(self.callbacks)

        self.windowHandle = Window(self.appName)
        self.frameCapturer = FrameCapturer()

    def run(self):
        self.isAppRunning = True

        while self.isAppRunning:
            self.cblist.on_loop_begin()

            frame = self.frameCapturer.CaptureFrame()
            self.windowHandle.ShowFrame(frame)

            self.cblist.on_loop_end()

        self.frameCapturer.Release()

        Window.KillAllWindows()

