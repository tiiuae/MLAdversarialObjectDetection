"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: March 29, 2022

Purpose: project images over checkerboard
"""
import cv2
import numpy as np

BOARD_SIZE = 7, 7


def project(img):
    _, corners = cv2.findChessboardCorners(img, BOARD_SIZE)
    return corners


def main():
    from streaming import Stream
    stream = Stream()
    for frame in stream.playing():
        print(project(frame), 'done')
        cv2.imshow('Frame', frame.astype(np.uint8))
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
