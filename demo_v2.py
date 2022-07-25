"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: July 13, 2022

Purpose: demonstrate the adversarial patch attack on person detection
"""
import ast
import logging
import os.path

import cv2
import numpy as np

import adv_patch
import util

util.allow_direct_imports_from('automl/efficientdet')

import detector
import generator
import streaming

logger = util.get_logger(__name__, logging.DEBUG)

SCORE_THRESH = .55


class Demo:
    txt_kwargs = {'font': cv2.FONT_HERSHEY_TRIPLEX, 'font_scale': .6, 'font_color': (0, 0, 0), 'thickness': 1,
                  'line_type': 1}

    def __init__(self, name, dct: detector.Detector, *, min_score_thresh=SCORE_THRESH):
        self.name = name
        self.dct_obj = dct
        self.min_score_thresh = min_score_thresh
        self._queue = []

    def measure_mean_score(self, sc):
        max_sc = max(sc) if len(sc) else 0.
        self._queue.append(max_sc)
        return round(np.mean(self._queue) * 100.)

    def run(self, frame):
        bb, sc = self.dct_obj.infer(frame)

        bb, sc = util.filter_by_thresh(bb, sc, self.min_score_thresh)
        frame = util.draw_boxes(frame, bb, sc)
        frame = cv2.rectangle(frame, (0, 0), (380, 30), (200, 200, 200), -1)
        util.puttext(frame, f'person detector avg. confidence: {self.measure_mean_score(sc)}%', (10, 20), **self.txt_kwargs)
        return frame, bb, max(sc) * 100. if len(sc) else 0.


class AttackDemo(Demo):
    def __init__(self, patch: adv_patch.AdversarialPatch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_obj = patch

    # noinspection PyMethodOverriding
    def run(self, frame, bb):
        mal_frame = self.patch_obj.add_adv_to_img(frame, bb)
        mal_frame, _, sc = super().run(mal_frame)
        return mal_frame, sc


class RecoveryDemo(Demo):
    def __init__(self, weights_file, patch: adv_patch.AdversarialPatch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_obj = patch
        self.config = self.dct_obj.driver.model.config
        self._antipatch = generator.define_generator(self.config.image_size, generator.NoiseGenerator)
        self._antipatch.load_weights(weights_file)
        self.atk_detection_thresh = 10.
        self.color1 = np.array([0, 0, 0])
        # self.color2 = np.array([0, 0, 0])

    def decay_colors(self):
        color1 = self.color1.astype(float) * .9
        # color2 = self.color2.astype(float) * .9
        self.color1 = color1.astype(int)
        # self.color2 = color2.astype(int)

    # noinspection PyMethodOverriding
    def run(self, frame, bb, sc, osc):
        mal_frame = self.patch_obj.add_adv_to_img(frame, bb)
        recovered_frame = np.clip(self.serve(mal_frame[np.newaxis]), 0., 255.).astype('uint8')
        recovered_frame, _, sc_after = super().run(recovered_frame)
        thresh = int(self.min_score_thresh * 100.)

        print(osc, sc, sc_after)
        if osc > thresh:
            score_recovery = sc_after - sc
            if score_recovery > self.atk_detection_thresh:
                self.color1 = np.array([255, 0, 0])
                # if sc_after > thresh:
                #     self.color2 = np.array([255, 0, 0])
            else:
                self.decay_colors()

        recovered_frame = cv2.rectangle(recovered_frame, (10, frame.shape[0] - 40), (40, frame.shape[0] - 10),
                                        tuple(self.color1.tolist()), -1)
        util.puttext(recovered_frame, 'attack detected', (50, frame.shape[0] - 10), **self.txt_kwargs)

        # recovered_frame = cv2.rectangle(recovered_frame, (170, frame.shape[0] - 40), (200, frame.shape[0] - 10),
        #                                 tuple(self.color2.tolist()), -1)
        # util.puttext(recovered_frame, 'recovery', (210, frame.shape[0] - 10), **self.txt_kwargs)
        return recovered_frame

    def serve(self, image_array):
        _, h, w, _ = image_array.shape
        image_array, scale = self.dct_obj.driver._preprocess(image_array)
        outputs = np.clip(2. * self._antipatch.predict(image_array) + image_array, -1., 1.)
        outputs *= self.config.stddev_rgb
        outputs += self.config.mean_rgb
        output = outputs[0]
        oh, ow, _ = output.shape
        dsize = int(ow * scale), int(oh * scale)
        output = cv2.resize(output, dsize)
        output = output[:h, :w]
        return output


def write_frame(frame, video_writer):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame)


def sort_func(string: str):
    i = string.index('preview') + len('preview')
    return int(string[i:-len('.png')])


def main(save_dir, input_file=None):
    save_dir = util.ensure_empty_dir(save_dir)

    stream = streaming.Stream(path=input_file, sort_func=sort_func, set_width=1280)
    config_override = {'nms_configs': {'iou_thresh': .5, 'score_thresh': 0.}}
    dct = detector.Detector(params=config_override, download_model=False)
    atk_dir = 'save_dir_new_data/patch_434_2.1692'

    with open(os.path.join(atk_dir, 'scale.txt')) as f:
        scale = ast.literal_eval(f.read())

    patch = adv_patch.AdversarialPatch(scale=scale, patch_file=os.path.join(atk_dir, 'patch.png'))

    player = stream.play()
    frame = next(player)
    output_size = frame.shape[1], frame.shape[0]

    demo_clean = Demo('clean', dct)
    demo_patch = AttackDemo(patch, 'adv. patch', dct)
    demo_recovery = RecoveryDemo('save_dir_def_imp/patch_143_0.0399/antipatch.h5', patch, 'recovery', dct)

    save_file_clean = os.path.join(save_dir, 'clean_v2.mp4')
    save_file_adv = os.path.join(save_dir, 'adv_v2.mp4')
    save_file_det = os.path.join(save_dir, 'det_v2.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_clean = cv2.VideoWriter(save_file_clean, fourcc, 18.0, output_size)
    out_adv = cv2.VideoWriter(save_file_adv, fourcc, 18.0, output_size)
    out_det = cv2.VideoWriter(save_file_det, fourcc, 18.0, output_size)

    try:
        for i, frame in enumerate(player):
            print(i, end=' ')
            ann_frame, bb, sc = demo_clean.run(frame.copy())
            mal_frame, sc_atk = demo_patch.run(frame.copy(), bb)
            dct_frame = demo_recovery.run(frame, bb, sc_atk, sc)

            # cv2.imshow('Frame', cv2.cvtColor(dct_frame, cv2.COLOR_RGB2BGR))
            #
            # # Press Q on keyboard to  exit
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            write_frame(ann_frame, out_clean)
            write_frame(mal_frame, out_adv)
            write_frame(dct_frame, out_det)
    finally:
        out_clean.release()
        out_adv.release()
        out_det.release()
        player.close()


if __name__ == '__main__':
    main(input_file='pics/demo_input.mp4',  # change to a mp4 file, or directory containing frames or None for webcam
         save_dir='q2_demos')