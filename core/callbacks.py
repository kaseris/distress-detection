from os import system, name
from time import time

import cv2


class Callback:

    def __init__(self):
        pass

    def on_loop_begin(self):
        raise NotImplementedError("on_loop_begin method must be implemented from a subclass")

    def on_loop_end(self):
        raise NotImplementedError("on_loop_begin method must be implemented from a subclass")

    def clear(self):
        if name == 'nt':
            _ = system('cls')
        else:
            _ = system('clear')


class CallbackList:

    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]

    def on_loop_begin(self):
        for cb in self.callbacks:
            cb.on_loop_begin()

    def on_loop_end(self):
        for cb in self.callbacks:
            cb.on_loop_end()


class TimeElapsedCallback(Callback):
    def __init__(self):
        super(TimeElapsedCallback, self).__init__()
        self.t0 = 0.0
        self.timeElapsed = 0.0

    def on_loop_begin(self):
        self.t0 = time()

    def on_loop_end(self):
        self.timeElapsed = time() - self.t0

        self.clear()
        print("Time elapsed: {:.3f}".format(self.timeElapsed))


class TerminateAppCallback(Callback):

    def __init__(self, app_instance):
        super(TerminateAppCallback, self).__init__()
        self.app_instance = app_instance

    def on_loop_begin(self):
        pass

    def on_loop_end(self):
        if cv2.waitKey(1) & 0xff == ord('q'):
            print("Terminating application.")
            self.app_instance.terminateApplicationLoop()
