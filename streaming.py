"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: March 28, 2022

Purpose: streaming module
"""
import cv2

import util

logger = util.get_logger(__name__)


class Stream:
    """stream from file or webcam"""
    def __init__(self, filename=None):
        self.filename = filename if filename is not None else 0
        self.cap = cv2.VideoCapture(self.filename)
        if not self.cap.isOpened():
            print('Error opening input video: {}'.format(filename))

    def play(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.error('failed to read frame')
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame

        self.cap.release()


def main():
    stream = Stream()
    for frame in stream.play():
        cv2.imshow('Frame', frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
