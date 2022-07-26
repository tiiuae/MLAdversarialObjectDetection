"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: July 16, 2022

Purpose: make frames from a directory into mp4 video. standalone script
"""
import cv2

import demo_v2
import streaming


def main(save_file, input_file=None):
    stream = streaming.Stream(path=input_file, sort_func=lambda x: int(x[:-len('.png')]), set_width=0)

    player = stream.play()
    frame = next(player)
    output_size = frame.shape[1], frame.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_file, fourcc, 24.0, output_size)

    try:
        for i, frame in enumerate(player):
            print(i+1)
            demo_v2.write_frame(frame, out)
    finally:
        out.release()
        player.close()


if __name__ == '__main__':
    main(input_file='pics/extracted_frames',  # change to directory containing frames
         save_file='pics/demo_input.mp4')
