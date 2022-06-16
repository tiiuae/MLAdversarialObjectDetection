"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 25, 2022

Purpose: demonstrate the adversarial patch attack on person detection
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
    txt_kwargs = {'font': cv2.FONT_HERSHEY_TRIPLEX, 'font_scale': .5, 'font_color': (148, 0, 211), 'thickness': 1,
                  'line_type': 1}

    def __init__(self, name, dct: detector.Detector, tw, th, *, min_score_thresh=SCORE_THRESH, rolling_avg_window=10):
        self.name = name
        self.dct_obj = dct
        self._queue = []
        self.min_score_thresh = min_score_thresh
        self._queue_length = rolling_avg_window
        self.title_pos = tw, th
        self.offset = 0

    def measure_mean_score(self, sc):
        max_sc = max(sc) if len(sc) else 0.
        self._add_item(max_sc)
        return round(np.mean(self._queue) * 100.)

    def _add_item(self, item):
        if len(self._queue) >= self._queue_length:
            del self._queue[0]
        self._queue.append(item)

    def run(self, frame):
        self.offset = 0
        title_pos_w, title_pos_h = self.title_pos
        self.offset += 30
        bb, sc = self.dct_obj.infer(frame)
        mean_sc = self.measure_mean_score(sc)

        bb, sc1 = util.filter_by_thresh(bb, sc, self.min_score_thresh)
        util.puttext(frame, self.name, self.title_pos, **self.txt_kwargs)
        util.puttext(frame, f'max detection score ({self._queue_length} frame mean.):'
                            f'{mean_sc}%', (title_pos_w, title_pos_h + self.offset), **self.txt_kwargs)
        frame = util.draw_boxes(frame, bb, sc1)
        return frame, bb, max(sc) * 100. if len(sc) else -1.


class AttackDemo(Demo):
    def __init__(self, patch: adv_patch.AdversarialPatch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_obj = patch

    def run(self, frame, bb):
        mal_frame = self.patch_obj.add_adv_to_img(frame, bb)
        mal_frame, _, sc = super().run(mal_frame)
        if sc > -1.:
            thresh = int(self.min_score_thresh * 100.)
            txt_kwargs = {k: v for k, v in self.txt_kwargs.items()}
            txt_kwargs.update(dict(font_color=(255, 0, 0) if sc < thresh else (0, 255, 0), font_scale=1.,
                                   thickness=2))
            detection = 'ATTACK SUCCESS' if sc < thresh else 'attack failed'
            title_pos_w, title_pos_h = self.title_pos
            self.offset -= 130
            util.puttext(mal_frame, detection, (title_pos_w + 80, title_pos_h + self.offset), **txt_kwargs)
        return mal_frame, sc


class RecoveryDemo(Demo):
    def __init__(self, weights_file, patch: adv_patch.AdversarialPatch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_obj = patch
        self.config = self.dct_obj.driver.model.config
        self._antipatch = generator.define_generator(self.config.image_size)
        self._antipatch.load_weights(weights_file)

    def run(self, frame, bb, sc):
        mal_frame = self.patch_obj.add_adv_to_img(frame, bb)
        recovered_frame = np.clip(self.serve(mal_frame[np.newaxis]), 0., 255.).astype('uint8')
        recovered_frame, _, sc_after = super().run(recovered_frame)
        if sc > -1.:
            score_recovery = sc_after - sc
            txt_kwargs = {k: v for k, v in self.txt_kwargs.items()}
            txt_kwargs.update(dict(font_color=(255, 0, 0) if score_recovery < 10. else (0, 255, 0), font_scale=1.,
                                   thickness=2))
            detection = 'no detection' if score_recovery < 10. else 'ATTACK DETECTED'
            title_pos_w, title_pos_h = self.title_pos
            self.offset -= 130
            util.puttext(recovered_frame, detection, (title_pos_w + 130, title_pos_h + self.offset), **txt_kwargs)
        return recovered_frame, sc_after

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


def make_graph(x, sc_clean, sc_random, sc_before, sc_after):
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
    stream = streaming.Stream(path=input_file)

    config_override = {'nms_configs': {'iou_thresh': .5, 'score_thresh': 0.},
                       'image_size': 480}
    dct = detector.Detector(params=config_override, download_model=False)
    atk_dir = 'save_dir/patch_12_0.5787'

    with open(os.path.join(atk_dir, 'scale.txt')) as f:
        # scale = ast.literal_eval(f.read())
        scale = .5

    patch = adv_patch.AdversarialPatch(scale=scale, patch_file=os.path.join(atk_dir, 'patch.png'))

    rand_patch = adv_patch.AdversarialPatch(scale=scale)

    player = stream.play()
    frame = next(player)
    output_size = 2 * frame.shape[1], 2 * frame.shape[0]

    demo_clean = Demo('clean', dct, 30, frame.shape[0] - 40)
    demo_patch = AttackDemo(patch, 'adv. patch', dct, 250, frame.shape[0] - 40)
    demo_rnd_patch = AttackDemo(rand_patch, 'random patch (as baseline)', dct, 30, frame.shape[0] - 40)
    demo_recovery = RecoveryDemo('det/save_dir/patch_100_0.7913/antipatch.h5', patch, 'recovery', dct, 30,
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

    # info_frame = make_info_frame(np.zeros_like(frame))
    x, a, b, c, d = [], [], [], [], []
    winsize = 30

    try:
        for i, frame in enumerate(player):
            ann_frame, bb, sc_clean = demo_clean.run(frame.copy())
            mal_frame, sc_before = demo_patch.run(frame.copy(), bb)
            ctrl_frame, sc_random = demo_rnd_patch.run(frame.copy(), bb)

            # frame = np.concatenate([ann_frame, mal_frame, ctrl_frame], axis=1)
            frame_top = np.concatenate([ann_frame, mal_frame], axis=1)
            dct_frame, sc_after = demo_recovery.run(frame, bb, sc_before)

            frame_bottom = np.concatenate([ctrl_frame, dct_frame], axis=1)
            frame = np.concatenate([frame_top, frame_bottom])

            x.append(i + 1)
            a.append(sc_clean)
            b.append(sc_random)
            c.append(sc_before)
            d.append(sc_after)
            x, a, b, c, d = x[-winsize:], a[-winsize:], b[-winsize:], c[-winsize:], d[-winsize:]
            data = make_graph(x, a, b, c, d)

            frame = cv2.resize(frame, output_size)
            center = frame.shape[0] // 2, frame.shape[1] // 2
            width, height = 400, 300
            half_width, half_height = width // 2, height // 2
            frame[center[0] - half_height:center[0] + half_height,
            center[1] - half_width:center[1] + half_width] = cv2.resize(data, (width, height))

            if live:
                ax.set_data(frame)
                plt.ion()
                plt.pause(.01)

            if save_file is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
    finally:
        if save_file is not None:
            out.release()
        player.close()


if __name__ == '__main__':
    main(input_file='eduardo_flying',  # change to a mp4 file or None for webcam stream
         save_file='out1.mp4',  # change to a mp4 file or None for no save
         live=True  # True if wish to see live streamq
         )
