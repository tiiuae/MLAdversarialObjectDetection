"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: July 14, 2022

Purpose: filter visDrone by person
"""
import pathlib

import cv2
from PIL import Image

import util


def main():
    video_filename = 'pics/IMG_0793.MOV'
    tgt_dir = util.ensure_empty_dir(pathlib.Path('pics/extracted_frames'))
    cap = cv2.VideoCapture(video_filename)
    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            frame_count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print('\rprocessing frame', frame_count, end='')
            Image.fromarray(frame).save(tgt_dir.joinpath(f'{frame_count}.png'))
    finally:
        cap.release()


if __name__ == '__main__':
    main()
