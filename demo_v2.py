"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: July 13, 2022

Purpose: designed v2 for q2 demo. they needed separate video results with minimal or no texts and graphs to accompany
them. this module has the same functionality as the demo.py module except the video files are separate and with minimal
text overlay
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
    """demo superclass"""
    txt_kwargs = {'font': cv2.FONT_HERSHEY_TRIPLEX, 'font_scale': .6, 'font_color': (0, 0, 0), 'thickness': 1,
                  'line_type': 1}

    def __init__(self, name, dct: detector.Detector, *, min_score_thresh=SCORE_THRESH):
        """
        init
        :param name: name of the demo
        :param dct: detector object to represent object detection model
        :param min_score_thresh: score threshold
        """
        self.name = name
        self.dct_obj = dct
        self.min_score_thresh = min_score_thresh
        self._queue = []

    def measure_mean_score(self, sc):
        """
        calculate mean of max score per image across all detections across all images seen
        :param sc: array of scores for an image
        :return: mean of max score per image between 0 and 100
        """
        max_sc = max(sc) if len(sc) else 0.
        self._queue.append(max_sc)
        return round(np.mean(self._queue) * 100.)

    def run(self, frame):
        """
        run demo for given video frame
        :param frame: frame
        :return: frame with bounding boxes and text overlaid, bounding boxes, and maximum score in the frame
        """
        bb, sc = self.dct_obj.infer(frame)

        bb, sc = util.filter_by_thresh(bb, sc, self.min_score_thresh)
        frame = util.draw_boxes(frame, bb, sc)
        frame = cv2.rectangle(frame, (0, 0), (380, 30), (200, 200, 200), -1)
        util.puttext(frame, f'person detector avg. confidence: {self.measure_mean_score(sc)}%', (10, 20), **self.txt_kwargs)
        return frame, bb, max(sc) * 100. if len(sc) else 0.


class AttackDemo(Demo):
    """attack demo subclass"""

    def __init__(self, patch: adv_patch.AdversarialPatch, *args, **kwargs):
        """
        init
        :param patch: AdversarialPatch object to represent the patch and its manipulations
        :param args: super args
        :param kwargs: super kwargs
        """
        super().__init__(*args, **kwargs)
        self.patch_obj = patch

    # noinspection PyMethodOverriding
    def run(self, frame, bb):
        """
        run this demo on a given video frame
        :param frame: frame
        :param bb: bounding box of persons needed where to attack
        :return: attacked frame, maximum score on this frame after the attack
        """
        mal_frame = self.patch_obj.add_adv_to_img(frame, bb)
        mal_frame, _, sc = super().run(mal_frame)
        return mal_frame, sc


class RecoveryDemo(Demo):
    """defender demo subclass"""

    def __init__(self, weights_file, patch: adv_patch.AdversarialPatch, *args, **kwargs):
        """
        init
        :param weights_file: defender model weight file - h5
        :param patch: AdversarialPatch object to represent the patch and its manipulations
        :param args: super args
        :param kwargs: super kwargs
        """
        super().__init__(*args, **kwargs)
        self.patch_obj = patch
        self.config = self.dct_obj.driver.model.config
        self._antipatch = generator.define_model(self.config.image_size, generator.PatchNeutralizer)
        self._antipatch.load_weights(weights_file)
        self.atk_detection_thresh = 10.
        self.flash_color = np.array([0, 0, 0])

    def decay_color(self):
        """decay the flash color every frame by 10 percent intensity"""
        flash_color = self.flash_color.astype(float) * .9
        self.flash_color = flash_color.astype(int)

    # noinspection PyMethodOverriding
    def run(self, frame, bb, sc, osc):
        """
        run this demo for a given video frame
        :param frame: frame
        :param bb: person bounding box where to apply the attack
        :param sc: object detection score after the attack
        :param osc: object detection score before the attack
        :return: recovered frame from the attack
        """
        mal_frame = self.patch_obj.add_adv_to_img(frame, bb)
        recovered_frame = np.clip(self.serve(mal_frame[np.newaxis]), 0., 255.).astype('uint8')
        recovered_frame, _, sc_after = super().run(recovered_frame)
        thresh = int(self.min_score_thresh * 100.)

        print(osc, sc, sc_after)
        if osc > thresh:
            score_recovery = sc_after - sc
            if score_recovery > self.atk_detection_thresh:
                self.flash_color = np.array([255, 0, 0])
            else:
                self.decay_color()

        recovered_frame = cv2.rectangle(recovered_frame, (10, frame.shape[0] - 40), (40, frame.shape[0] - 10),
                                        tuple(self.flash_color.tolist()), -1)
        util.puttext(recovered_frame, 'attack detected', (50, frame.shape[0] - 10), **self.txt_kwargs)
        return recovered_frame

    def serve(self, image_array):
        """
        run the defender model to restore the attacked areas in the image
        :param image_array: array of 1 image to send to the model
        :return: attack recovered image
        """
        _, h, w, _ = image_array.shape
        image_array, scale = self.dct_obj.driver._preprocess(image_array)
        outputs = np.clip(2. * self._antipatch.predict(image_array) + image_array, -1., 1.)
        outputs *= self.config.stddev_rgb
        outputs += self.config.mean_rgb
        output = outputs[0]
        oh, ow, _ = output.shape
        dsize = int(ow * scale), int(oh * scale)
        output = cv2.resize(output, dsize)

        # crop out grey area
        output = output[:h, :w]
        return output


def write_frame(frame, video_writer):
    """
    write single frame to a given video writer
    :param frame: frame
    :param video_writer: opencv video writer
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame)


def sort_func(string: str):
    """
    specific sorting key function for a certain directory containing images in a certain format.
    :param string: filename
    :return: sorting key
    """
    i = string.index('preview') + len('preview')
    return int(string[i:-len('.png')])


def main(save_dir, input_file=None):
    """
    run all demos
    :param save_dir: save directory to save all video files in
    :param input_file: input mp4 video file or None for webcam stream or a directory containing images
    """
    # ensure clean output dir
    save_dir = util.ensure_empty_dir(save_dir)

    # init steam class for input flow
    stream = streaming.Stream(path=input_file, sort_func=sort_func, set_width=1280)

    # init object detection model and configs
    config_override = {'nms_configs': {'iou_thresh': .5, 'score_thresh': 0.}}
    dct = detector.Detector(params=config_override, download_model=False)

    # load learned adv. patch and scale of patch relative to person bounding box length
    atk_dir = 'save_dir_new_data/patch_434_2.1692'
    with open(os.path.join(atk_dir, 'scale.txt')) as f:
        scale = ast.literal_eval(f.read())

    # define patch manipulator object for the adv. patch
    patch = adv_patch.AdversarialPatch(scale=scale, patch_file=os.path.join(atk_dir, 'patch.png'))

    # init stream player
    player = stream.play()

    # determine output frame size
    frame = next(player)
    output_size = frame.shape[1], frame.shape[0]

    # init demo objects
    demo_clean = Demo('clean', dct)
    demo_patch = AttackDemo(patch, 'adv. patch', dct)
    demo_recovery = RecoveryDemo('save_dir_def_imp/patch_143_0.0399/antipatch.h5', patch, 'recovery', dct)

    # init save files and writers
    save_file_clean = os.path.join(save_dir, 'clean_v2.mp4')
    save_file_adv = os.path.join(save_dir, 'adv_v2.mp4')
    save_file_det = os.path.join(save_dir, 'det_v2.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_clean = cv2.VideoWriter(save_file_clean, fourcc, 18.0, output_size)
    out_adv = cv2.VideoWriter(save_file_adv, fourcc, 18.0, output_size)
    out_det = cv2.VideoWriter(save_file_det, fourcc, 18.0, output_size)

    try:
        # play
        for i, frame in enumerate(player):
            print(i, end=' ')
            ann_frame, bb, sc = demo_clean.run(frame.copy())
            mal_frame, sc_atk = demo_patch.run(frame.copy(), bb)
            dct_frame = demo_recovery.run(frame, bb, sc_atk, sc)

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
