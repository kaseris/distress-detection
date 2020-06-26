import os
import sys

import numpy as np

from .callbacks import CallbackList
from .callbacks import TimeElapsedCallback, TerminateAppCallback, ProgressBarCallback
from .image_processor import ImageProcessor

from ..frame_capturer import FrameCapturer
from ..window import Window



class Application:
    """
    TODO: Initialise the callback list from the base class.
    """
    def __init__(self, app_name):
        self.appName = app_name
        self.isAppRunning = False

    def __del__(self):
        sys.stdout.write('\nDestroying %s' % (self.appName))
        sys.stdout.flush()

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


class DatasetApplication(Application):

    def __init__(self, app_name, dataset_dir, output_dir, samples_per_loop=15, steps=None, callbacks=None, out_file_extension=".jpg"):
        super(DatasetApplication, self).__init__(app_name)
        self.samples_per_loop = samples_per_loop
        self.extension = out_file_extension
        self.dataset_dir = dataset_dir
        self.total_elements = len(os.listdir(self.dataset_dir))
        self.output_dir = output_dir
        self.steps = steps or round((self.total_elements / self.samples_per_loop))
        self.curr_step = 0

        self.callbacks = callbacks or []
        self.callbacks.append(ProgressBarCallback(self))
        self.cblist = CallbackList(self.callbacks)

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def summary(self):
        print(f"Application name: {self.appName}")
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total samples: {self.total_elements}")
        print(f"Total batches: {int(self.total_elements / self.samples_per_loop)}")

    def run(self):

        while self.curr_step < self.steps:
            img_names = []
            img_paths = []
            img_names = os.listdir(self.dataset_dir)[self.curr_step * self.samples_per_loop:(self.curr_step + 1) * self.samples_per_loop]
            img_paths = [self.dataset_dir + "/" + name for name in img_names]

            for img_path, name in zip(img_paths, img_names):
                img = ImageProcessor.ReadImage(img_path)
                a = (img.shape[0] <= 256) or (img.shape[1] <= 256)
                if a:
                    continue
                random_crops = np.random.randint(0, min(img.shape[0] - 256, img.shape[1] - 256), (4, 2))

                for i, crop in enumerate(random_crops.tolist()):
                    cropped = ImageProcessor.Crop(img, 256, 1, crop)
                    ImageProcessor.StoreImage(self.output_dir + "/" + name + f"_cropped{i}" + self.extension, cropped)

                cropped = ImageProcessor.Crop(img, 256)
                ImageProcessor.StoreImage(self.output_dir + "/" + name + f"_cropped_center" + self.extension, cropped)

            self.curr_step += 1
            self.cblist.on_loop_end()
