import sys
from os import system, name
from time import time

import cv2
import psutil


class Callback:

    def __init__(self):
        pass

    def on_loop_begin(self):
        raise NotImplementedError("on_loop_begin method must be implemented from a subclass")

    def on_loop_end(self):
        raise NotImplementedError("on_loop_begin method must be implemented from a subclass")

    @staticmethod
    def clear():
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
        Callback.clear()
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

        print("Time elapsed: {:.3f}ms".format(self.timeElapsed * 1000))


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

class HardwareStatsCallback(Callback):
    def __init__(self):
        super(HardwareStatsCallback, self).__init__()

    def get_size(self, bytes, suffix="B"):
        factor=1024
        for unit in ["", "K", "M", "G", "T"]:
            if bytes < factor:
                return f"{bytes:.2f}{unit}{suffix}"
            bytes /= factor

    def on_loop_begin(self):
        pass

    def on_loop_end(self):

        # Print CPU usage
        print("="*40, "CPU Info", "="*40)
        print("Per core CPU usage:")
        for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=0.0001)):
            print(f"Core: {i}: {percentage}%")
        print(f"Total CPU usage: {psutil.cpu_percent()}%")


class ProgressBarCallback(Callback):
    """

    https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    """
    def __init__(self, app_instance):
        super(ProgressBarCallback, self).__init__()
        self.app_instance = app_instance
        self.bar_len = 50

    def on_loop_begin(self):
        pass

    def on_loop_end(self):
        filled_len = int(round(self.bar_len * self.app_instance.curr_step / float(self.app_instance.steps)))

        percent = round(100 * self.app_instance.curr_step / float(self.app_instance.steps), 1)
        bar = '=' * filled_len + '-' * (self.bar_len - filled_len)

        sys.stdout.write('Performing crop on the dataset images\n')
        sys.stdout.write('[%s] Step: %s/%s %s%s\r' % (bar, self.app_instance.curr_step, self.app_instance.steps, percent, '%'))
        sys.stdout.flush()
