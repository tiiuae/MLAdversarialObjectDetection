"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 25, 2022

Purpose: demonstrate the adversarial patch attack on person detection
"""
import ast
import logging

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import adv_patch
import detector
import streaming
import util

logger = util.get_logger(__name__, logging.DEBUG)

SCORE_THRESH = .5


class Demo:

    txt_kwargs = {'font_scale': .6, 'font_color': (255, 0, 0), 'thickness': 2, 'line_type': 1}

    def __init__(self, name, dct: detector.Detector, *, min_score_thresh=SCORE_THRESH, rolling_avg_window=10):
        self.name = name
        self.dct_obj = dct
        self._queue = []
        self.min_score_thresh = min_score_thresh
        self._queue_length = rolling_avg_window

    def measure_mean_score(self, sc):
        max_sc = max(sc) if len(sc) else 0.
        self._add_item(max_sc)
        return round(np.mean(self._queue) * 100.)

    def _add_item(self, item):
        if len(self._queue) >= self._queue_length:
            del self._queue[0]
        self._queue.append(item)

    def run(self, frame):
        title_pos = title_pos_h, title_pos_w = 30, 30
        offset = 30
        bb, sc = self.dct_obj.infer(frame)
        util.puttext(frame, self.name, title_pos, **self.txt_kwargs)
        util.puttext(frame, f'max detection score ({self._queue_length} frame mean.):'
                            f'{self.measure_mean_score(sc)}%',
                     (title_pos_w, title_pos_h + offset), **self.txt_kwargs)

        bb, sc = util.filter_by_thresh(bb, sc, self.min_score_thresh)
        frame = util.draw_boxes(frame, bb, sc)
        return frame, bb, sc


class AttackDemo(Demo):
    def __init__(self, patch: adv_patch.AdversarialPatch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_obj = patch

    def run(self, frame, bb):
        mal_frame = self.patch_obj.add_adv_to_img(frame, bb)
        mal_frame, _, _ = super().run(mal_frame)
        return mal_frame


def make_info_frame(info_frame):
    title_pos_h, title_pos_w = 0, 10

    def write(note, w_offset=0, h_offset=30):
        nonlocal title_pos_w, title_pos_h
        title_pos_h += h_offset
        title_pos_w += w_offset
        util.puttext(info_frame, note, (title_pos_w, title_pos_h), **Demo.txt_kwargs)

    thresh = round(SCORE_THRESH * 100)
    write('Note:')
    write('Step 1: clean pass through image')
    write('Step 2: on output of step 1, and on persons with scores more')
    write(f'than {thresh}%:')
    write('a: attack with pretrained adv. patch (100 epochs on COCO)', w_offset=30)
    write('b: attack with random adv. patch')
    write('Step 3: in each case, scores are calculated as:', w_offset=-30)
    write('mean(max score per frame before thresholding) over last', w_offset=30)
    write('10 frames seen')
    write(f'Step 4: threshold and draw bounding boxes if score >= {thresh}%', w_offset=-30)
    return info_frame


def main(input_file=None, save_file=None, live=False):
    stream = streaming.Stream(filename=input_file)
    dct = detector.Detector(download_model=False)

    patch = adv_patch.AdversarialPatch(patch_file='save_dir/patch_50_0.7146.tiff')

    with open('save_dir/patch_70_0.7369/scale.txt') as f:
        scale = ast.literal_eval(f.read())

    rand_patch = adv_patch.AdversarialPatch(scale=scale)

    demo_clean = Demo('clean', dct)
    demo_patch = AttackDemo(patch, 'adv. patch', dct)
    demo_rnd_patch = AttackDemo(rand_patch, 'random patch (as baseline)', dct)

    player = stream.play()
    frame = next(player)
    output_size = 2 * frame.shape[1], 2 * frame.shape[0]

    if live:
        matplotlib.use("TkAgg")
        plt.rcParams['toolbar'] = 'None'

        scale = 15
        ratio = output_size[1] / output_size[0]

        fig, ax = plt.subplots(figsize=(scale, scale * ratio))
        ax.axis('off')
        ax = ax.imshow(np.zeros((output_size[1], output_size[0], 3)))
        fig.subplots_adjust(0, 0, 1, 1)
        # fig.canvas.manager.window.overrideredirect(1)
        plt.ion()
        plt.show()

    if save_file is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_file, fourcc, 8.0, output_size)

    try:
        for frame in player:
            ann_frame, bb, sc = demo_clean.run(frame.copy())
            mal_frame = demo_patch.run(frame.copy(), bb)
            ctrl_frame = demo_rnd_patch.run(frame, bb)

            # frame = np.concatenate([ann_frame, mal_frame, ctrl_frame], axis=1)
            frame_top = np.concatenate([ann_frame, mal_frame], axis=1)
            info_frame = make_info_frame(np.zeros_like(ctrl_frame))
            frame_bottom = np.concatenate([ctrl_frame, info_frame], axis=1)
            frame = np.concatenate([frame_top, frame_bottom])
            frame = cv2.resize(frame, output_size)

            if live:
                ax.set_data(frame)
                plt.pause(.01)

            if save_file is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
    finally:
        if save_file is not None:
            out.release()
        player.close()


if __name__ == '__main__':
    main(input_file='production ID 5058322.mp4',  # change to a mp4 file or None for webcam stream
         save_file='out4.mp4',  # change to a mp4 file or None for no save
         live=True  # True if wish to see live stream
         )
