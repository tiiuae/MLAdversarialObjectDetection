"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: March 28, 2022

Purpose: streaming module
"""
import os
import time

import numpy as np
from PIL import Image
import cv2

import util

logger = util.get_logger(__name__)


class Stream:
    """stream from file or webcam"""
    def __init__(self, path=None, *, set_width=640):
        self.path = path if path is not None else 0
        self.set_width = set_width

        if not path or os.path.isfile(path):
            self.cap = cv2.VideoCapture(self.path)
            if not self.cap.isOpened():
                print('Error opening input video: {}'.format(path))

    def play_from_video(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    logger.info('end of steam')
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = frame.shape
                scale = self.set_width / w
                h *= scale
                frame = cv2.resize(frame, (self.set_width, int(h)))
                yield frame
        finally:
            self.cap.release()

    def play_from_dir(self):
        files = os.listdir(self.path)
        files.sort(key=lambda x: int(x[len('preview'):-len('.png')]))
        for file in files:
            time.sleep(1/24)
            file = os.path.join(self.path, file)
            frame = np.asarray(Image.open(file).convert('RGB'))
            h, w, _ = frame.shape
            scale = self.set_width / w
            h *= scale
            frame = cv2.resize(frame, (self.set_width, int(h)))
            yield frame

    def play(self):
        if os.path.isdir(self.path):
            yield from self.play_from_dir()
        else:
            yield from self.play_from_video()


def main():
    stream = Stream(path='eduardo_flying')
    for frame in stream.play():
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Frame', frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
