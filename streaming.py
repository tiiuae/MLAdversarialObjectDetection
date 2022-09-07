"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: March 28, 2022

Purpose: streaming module. Useful for inference time. This module can take an input source and stream it downstream for
further processing. It can read videos, webcam or directories containing images for streaming
"""
import os
import time

import numpy as np
from PIL import Image
import cv2

import util

logger = util.get_logger(__name__)


class Stream:
    """stream from file, directory or webcam"""

    def __init__(self, path=None, *, filter_func=None, sort_func=None, set_width=640):
        """
        init
        :param path: may be a video file, directory containing input images or webcam is left None
        :param filter_func: only used when path is a directory containing images. must be a callable which produces a
        boolean for each file passed. files for which this function returns false, they will be omitted during streaming
        example usage could be to only steam images of certain type or certain quality and to exclude non image files in
        the same directory
        :param sort_func: only used when path is a directory containing images. must be a callable. it is used to sort
        the order of stream from directory when an image ordering is needed
        :param set_width: whether to resize image to this width during streaming, height will be adjusted to maintain
        aspect ratio, set_width=0 will mean no rescaling.
        """
        self.path = path = path if path is not None else 0
        self.set_width = set_width

        if not path or os.path.isfile(path):
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                print('Error opening input video: {}'.format(path))
        elif os.path.isdir(path):
            self.files = os.listdir(path)
            if filter_func:
                logger.info('filtering dataset by label constraints...')
                self.files = list(filter(filter_func, self.files))
                logger.info(f'done. data size is {len(self.files)}')
            if sort_func:
                self.files.sort(key=sort_func)

    def change_frame_size(self, frame):
        """
        rescaling functionality. maintains aspect ratio
        :param frame: frame
        :return: resized frame
        """
        h, w, _ = frame.shape
        scale = self.set_width / w
        h *= scale
        return cv2.resize(frame, (self.set_width, int(h)))

    def play_from_video(self):
        """
        stream player. a generator function to decode a video and yield frame by frame
        :yield: single frame
        """
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    logger.info('end of steam')
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.set_width:
                    frame = self.change_frame_size(frame)
                yield frame
        finally:
            self.cap.release()

    def play_from_list(self):
        """
        stream player. a generator function to yield frames from a list of image files (obtained from a directory)
        :yield: single frame
        """
        for file in self.files:
            time.sleep(1/24)
            file = os.path.join(self.path, file)
            frame = np.asarray(Image.open(file).convert('RGB'))
            if self.set_width:
                frame = self.change_frame_size(frame)
            yield frame

    def play(self):
        """
        stream player. a generator function combine the functionality of other generators
        :yield: single frame
        """
        if os.path.isdir(self.path):
            yield from self.play_from_list()
        else:
            yield from self.play_from_video()


def test():
    """test only"""
    stream = Stream(path='eduardo_flying')
    for frame in stream.play():
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Frame', frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    test()
