"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 25, 2022

Purpose: demonstrate the adversarial patch attack on person detection and its detection and recovery mechanism as a
single video output file containing graph and text overlays
"""
import ast
import logging
import os.path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import adv_patch
import detector
import generator
import streaming
import util

logger = util.get_logger(__name__, logging.DEBUG)

SCORE_THRESH = .55


class Demo:
    """demo superclass"""

    txt_kwargs = {'font': cv2.FONT_HERSHEY_TRIPLEX, 'font_scale': .7, 'font_color': (0, 0, 255), 'thickness': 1,
                  'line_type': 1}

    def __init__(self, name, dct: detector.Detector, tw, th, *, min_score_thresh=SCORE_THRESH):
        """
        init
        :param name: name of the demo
        :param dct: detector object to represent object detection model
        :param tw: text location y to place text in video
        :param th: text location x to place text in video
        :param min_score_thresh: score threshold
        """
        self.name = name
        self.dct_obj = dct
        self._sc_arr = []
        self.min_score_thresh = min_score_thresh
        self.title_pos = tw, th
        self.offset = 0

    def measure_mean_score(self, sc):
        """
        calculate mean of max score per image across all detections across all images seen
        :param sc: array of scores for an image
        :return: mean of max score per image between 0 and 100
        """
        max_sc = max(sc) if len(sc) else 0.
        self._sc_arr.append(max_sc)
        return round(np.mean(self._sc_arr) * 100.)

    def run(self, frame):
        """
        run demo for given video frame
        :param frame: frame
        :return: frame with bounding boxes and text overlaid, bounding boxes, and maximum score in the frame
        """
        self.offset = 0
        title_pos_w, title_pos_h = self.title_pos
        self.offset += 30

        # call model inference
        bb, sc = self.dct_obj.infer(frame)
        mean_sc = self.measure_mean_score(sc)

        # draw stuff on frame
        bb, sc1 = util.filter_by_thresh(bb, sc, self.min_score_thresh)
        util.puttext(frame, self.name, self.title_pos, **self.txt_kwargs)
        util.puttext(frame, f'average obj. detection score:'
                            f'{mean_sc}%', (title_pos_w, title_pos_h + self.offset), **self.txt_kwargs)
        frame = util.draw_boxes(frame, bb, sc1)
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
        self._atk_queue = []

    def calc_asr(self):
        """
        calculate attack success rate across all the images seen so far
        :return: asr in 0 to 100 range
        """
        thresh = int(self.min_score_thresh * 100.)
        success = len(list(filter(lambda x: x < thresh, self._atk_queue)))
        return round(success / len(self._atk_queue) * 100.)

    def run(self, frame, bb, osc):
        """
        run this demo on a given video frame
        :param frame: frame
        :param bb: bounding box of persons needed where to attack
        :param osc: original score (from clean image)
        :return: attacked frame, maximum score on this frame after the attack
        """
        # attack image
        mal_frame = self.patch_obj.add_adv_to_img(frame, bb)

        # call object detection model
        mal_frame, _, sc = super().run(mal_frame)

        # if original score was less than threshold then the clean image itself did not have any object else draw
        # bounding boxes
        thresh = int(self.min_score_thresh * 100.)
        if sc > -1. and osc > thresh:
            self._atk_queue.append(sc)
            util.puttext(mal_frame, f'attack success rate: {self.calc_asr()}%', (30, 30), **self.txt_kwargs)
            txt_kwargs = {k: v for k, v in self.txt_kwargs.items()}
            txt_kwargs.update(dict(font_color=(255, 0, 0) if sc < thresh else (0, 255, 0), font_scale=1.,
                                   thickness=1))
            detection = 'ATTACK SUCCESS' if sc < thresh else 'attack failed'
            title_pos_w, title_pos_h = self.title_pos
            self.offset -= 130
            util.puttext(mal_frame, detection, (title_pos_w + 80, title_pos_h + self.offset), **txt_kwargs)
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

        # defender model
        self._antipatch = generator.define_model(self.config.image_size, generator.PatchNeutralizer)
        self._antipatch.load_weights(weights_file)

        self._diff_queue = []
        self.atk_detection_thresh = 10.

    def calc_adr(self):
        """
        calculate attack detection rate across all the images seen so far
        :return: adr between 0 and 100 range
        """
        success = len(list(filter(lambda x: x > self.atk_detection_thresh, self._diff_queue)))
        return round(success / len(self._diff_queue) * 100.)

    def run(self, frame, bb, sc, osc):
        """
        run this demo for a given video frame
        :param frame: frame
        :param bb: person bounding box where to apply the attack
        :param sc: object detection score after the attack
        :param osc: object detection score before the attack
        :return: recovered frame from the attack, max object detection score after the recovery (defence)
        """
        # attack image
        mal_frame = self.patch_obj.add_adv_to_img(frame, bb)

        # run defender model on attacked image
        recovered_frame = np.clip(self.serve(mal_frame[np.newaxis]), 0., 255.).astype('uint8')

        # run object detection model on recovered image
        recovered_frame, _, sc_after = super().run(recovered_frame)

        # if original score was less than threshold then the clean image itself did not have any object else draw
        # bounding boxes
        thresh = int(self.min_score_thresh * 100.)
        if osc > thresh:
            score_recovery = sc_after - sc
            self._diff_queue.append(score_recovery)
            util.puttext(recovered_frame, f'attack detection rate: {self.calc_adr()}%', (180, 30), **self.txt_kwargs)
            txt_kwargs = {k: v for k, v in self.txt_kwargs.items()}
            txt_kwargs.update(dict(font_color=(255, 0, 0) if score_recovery < self.atk_detection_thresh
            else (0, 255, 0), font_scale=1., thickness=1))
            detection = 'no detection' if score_recovery < self.atk_detection_thresh else 'ATTACK DETECTED'
            title_pos_w, title_pos_h = self.title_pos
            self.offset -= 130
            util.puttext(recovered_frame, detection, (title_pos_w + 130, title_pos_h + self.offset), **txt_kwargs)
        return recovered_frame, sc_after

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

        # remove gray band after rescaling to original size
        output = output[:h, :w]
        return output


def make_graph(x, sc_clean, sc_random, sc_before, sc_after):
    """
    draw graph to put on the output video showing plots of respective object detection model scores, before attack,
    after attack and after attack recovery
    :param x: frame index
    :param sc_clean: scores before attack
    :param sc_random: scores after random patch attack
    :param sc_before: scores after adv. patch attack
    :param sc_after: scores after attack recovery
    :return: matplotlib figure as a numpy rgb image array (to be put on video)
    """
    x = np.array(x)
    sc_clean = np.array(sc_clean)
    sc_random = np.array(sc_random)
    sc_before = np.array(sc_before)
    sc_after = np.array(sc_after)
    plt.ioff()
    fig = plt.figure()
    fig.add_subplot(211)
    plt.plot(x, sc_clean, color='green', label='clean')
    plt.plot(x, sc_random, color='magenta', label='random')
    plt.plot(x, sc_before, color='red', label='attack')
    plt.plot(x, sc_after, color='blue', label='recovery')
    thresh = int(SCORE_THRESH * 100.)
    plt.plot(x, np.ones_like(x) * thresh, color='black', linestyle=':', label=f'obj detection thresh ({thresh})')
    delta = sc_after - sc_before
    mask = delta >= 10.
    x_det = x[mask]
    y_det = sc_after[mask]
    delta_det = delta[mask]
    if x_det.size:
        plt.scatter(x_det, y_det, color='blue', s=10)
    plt.legend(loc='lower left')
    plt.ylabel('max object detection score')
    plt.grid(True)

    fig.add_subplot(212)
    plt.plot(x, delta, color='brown', label='change in score (blue - red)')
    if x_det.size:
        plt.scatter(x_det, delta_det, color='brown', s=10)
    plt.plot(x, np.ones_like(x) * 10., color='black', linestyle=':', label='attack detection thresh (10)')
    plt.xlabel('frame number')
    plt.ylabel('delta score')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    plt.ion()
    return data


def main(input_file=None, save_file=None, live=False):
    """
    run demos and put the results into a single video with graphs and text overlay
    :param input_file: input mp4 video file or None for webcam stream or a directory containing images
    :param save_file: output mp4 video filename or None for no save
    :param live: whether to show live results on screen too as the output video is rendered or not
    """
    # init steam class for input flow
    stream = streaming.Stream(path=input_file, sort_func=lambda x: int(x[len('preview'):-len('.png')]))

    # init object detection model and configs
    config_override = {'nms_configs': {'iou_thresh': .5, 'score_thresh': 0.}}
    dct = detector.Detector(params=config_override, download_model=False)

    # load learned adv. patch and scale of patch relative to person bounding box length
    atk_dir = 'save_dir_new_data/patch_434_2.1692'
    with open(os.path.join(atk_dir, 'scale.txt')) as f:
        scale = ast.literal_eval(f.read())

    # define patch manipulator objects for the adv. and random patches
    patch = adv_patch.AdversarialPatch(scale=scale, patch_file=os.path.join(atk_dir, 'patch.png'))
    rand_patch = adv_patch.AdversarialPatch(scale=scale)

    # init stream player
    player = stream.play()

    # determine output frame size
    frame = next(player)
    output_size = 2 * frame.shape[1], 2 * frame.shape[0]

    # init demo objects
    demo_clean = Demo('clean', dct, 30, frame.shape[0] - 40)
    demo_patch = AttackDemo(patch, 'adv. patch', dct, 250, frame.shape[0] - 40)
    demo_rnd_patch = AttackDemo(rand_patch, 'random patch (as baseline)', dct, 30, frame.shape[0] - 40)
    demo_recovery = RecoveryDemo('det/save_dir_attention/patch_193_0.0484/antipatch.h5', patch, 'recovery', dct, 30,
                                 frame.shape[0] - 40)

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
        fig.show()

    if save_file is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_file, fourcc, 8.0, output_size)

    x, a, b, c, d = [], [], [], [], []
    winsize = 30

    try:
        # play frames
        for i, frame in enumerate(player):
            ann_frame, bb, sc_clean = demo_clean.run(frame.copy())
            mal_frame, sc_before = demo_patch.run(frame.copy(), bb, sc_clean)
            ctrl_frame, sc_random = demo_rnd_patch.run(frame.copy(), bb, sc_clean)

            frame_top = np.concatenate([ann_frame, mal_frame], axis=1)
            dct_frame, sc_after = demo_recovery.run(frame, bb, sc_before, sc_clean)

            frame_bottom = np.concatenate([ctrl_frame, dct_frame], axis=1)

            # make a single frame containing 4 concatenated results
            frame = np.concatenate([frame_top, frame_bottom])

            # make graph
            x.append(i + 1)
            a.append(sc_clean)
            b.append(sc_random)
            c.append(sc_before)
            d.append(sc_after)
            x, a, b, c, d = x[-winsize:], a[-winsize:], b[-winsize:], c[-winsize:], d[-winsize:]
            data = make_graph(x, a, b, c, d)

            frame = cv2.resize(frame, output_size)

            # put graph at the center of the video
            center = frame.shape[0] // 2, frame.shape[1] // 2
            width, height = 300, 250
            half_width, half_height = width // 2, height // 2
            frame[center[0] - half_height:center[0] + half_height,
            center[1] - half_width:center[1] + half_width] = cv2.resize(data, (width, height))

            if live:
                ax.set_data(frame)
                plt.ion()
                plt.pause(.1)

            if save_file is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
    finally:
        if save_file is not None:
            out.release()
        player.close()


if __name__ == '__main__':
    main(input_file='pics/demo_input.mp4',  # change to a mp4 file or None for webcam stream
         save_file='out111.mp4',  # change to a mp4 file or None for no save
         live=True  # True if wish to see live streaminq
         )
