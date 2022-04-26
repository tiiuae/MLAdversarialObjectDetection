"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 25, 2022

Purpose: demonstrate the adversarial patch attack on person detection
"""
import logging

import cv2
import numpy as np

import adv_patch
import detector
import streaming
import util

logger = util.get_logger(__name__, logging.DEBUG)


class Demo:
    def __init__(self, dct: detector.Detector, patch: adv_patch.AdversarialPatch = None, *, rolling_avg_window=10,
                 min_iou=.5, **txt_kwargs):
        self.patch_obj = patch
        self.dct_obj = dct
        self._queue = []
        self._queue_length = rolling_avg_window
        self._font_scale = txt_kwargs.get('font_scale')
        self._font_color = txt_kwargs.get('font_color')
        self._thickness = txt_kwargs.get('thickness')
        self._line_type = txt_kwargs.get('line_type')
        self._min_iou = min_iou

    def puttext(self, img, text, pos):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = pos
        cv2.putText(img, text,
                    bottom_left_corner_of_text,
                    font,
                    self._font_scale,
                    self._font_color,
                    self._thickness,
                    self._line_type)

    def measure_success(self, gt_boxes, boxes):
        if not len(gt_boxes):
            return 0.

        failures = sum(any(util.calculate_iou(box, gt_box) >= self._min_iou for gt_box in gt_boxes) for box in boxes)
        cur_success = (1. - failures / len(gt_boxes)) * 100.
        self._add_item(cur_success)
        return np.mean(self._queue)

    def _add_item(self, item):
        if len(self._queue) >= self._queue_length:
            del self._queue[0]
        self._queue.append(item)

    def attack(self, frame, bb):
        title_pos = title_pos_h, title_pos_w = 30, 30
        offset = 30
        mal_frame = self.patch_obj.add_adv_to_img(frame, bb)
        bb1, sc1 = self.dct_obj.infer(mal_frame)
        mal_frame = util.draw_boxes(mal_frame, bb1, sc1)
        self.puttext(mal_frame, 'adv. patch', title_pos)
        self.puttext(mal_frame, f'attack success (10 frame avg.):{self.measure_success(bb, bb1)}',
                     (title_pos_h, title_pos_w + offset))
        return mal_frame


def main(input_file=None, save_file=None, live=False):
    stream = streaming.Stream(filename=input_file)
    dct = detector.Detector(download_model=False, min_score_thresh=.5)

    patch = adv_patch.AdversarialPatch(patch_file='save_dir/patch_45_0.7042.tiff')
    rand_patch = adv_patch.AdversarialPatch()

    txt_kwargs = {'font_scale': .8, 'font_color': (0, 255, 0), 'thickness': 1, 'line_type': 1}

    demo_patch = Demo(dct, patch, **txt_kwargs)
    demo_rnd_patch = Demo(dct, rand_patch, **txt_kwargs)

    player = stream.play()
    frame = next(player)
    output_size = 2 * frame.shape[1] // 3, 2 * frame.shape[0] // 3

    if save_file is not None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(save_file, fourcc, 20.0, output_size)

    for frame in player:
        bb, sc = dct.infer(frame)
        logger.debug(f'{bb}, {sc}')

        mal_frame = demo_patch.attack(frame, bb)
        ctrl_frame = demo_rnd_patch.attack(frame, bb)

        frame = util.draw_boxes(frame, bb, sc)
        demo_patch.puttext(frame, 'clean', (30, 30))
        frame = cv2.cvtColor(np.concatenate([frame, mal_frame, ctrl_frame], axis=1), cv2.COLOR_RGB2BGR)

        frame = cv2.resize(frame, output_size)

        if save_file is not None:
            out.write(frame)

        if live:
            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if save_file is not None:
                    out.release()
                break

    if save_file is not None:
        out.release()


if __name__ == '__main__':
    main(input_file='pedestrian.mp4',  # change to a mp4 file or None for webcam stream
         save_file='out.mp4',  # change to a mp4 file or None for no save
         live=True  # True if wish to see live stream
         )
