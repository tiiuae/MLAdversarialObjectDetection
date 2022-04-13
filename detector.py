"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: March 28, 2022

Purpose: detection module
"""
import logging

import cv2
import numpy as np
import tensorflow as tf

import util
from automl.efficientdet.tf2 import infer_lib
from visualize.vis_utils import draw_bounding_box_on_image_array

MODEL = 'efficientdet-lite4'
logger = util.get_logger(__name__)


class Detector:
    """Inference with efficientDet object detector"""
    def __init__(self, *, download_model=False):
        if download_model:
            # Download checkpoint.
            util.download(MODEL)
            logger.info(f'Using model in {MODEL}')

        self.driver = infer_lib.KerasDriver(MODEL, debug=False, model_name=MODEL)

    def infer(self, frame, return_annotations_only=True, max_boxes=200, min_score_thresh=.5):
        raw_frames = np.array([frame])
        detections_bs = self.driver.serve(raw_frames)
        logger.debug([type(x) for x in detections_bs])
        boxes, scores, classes, _ = tf.nest.map_structure(np.array, detections_bs)
        boxes, scores, classes = boxes[0], scores[0], classes[0]

        n_boxes = 0
        bb, sc = [], []
        for i in range(boxes.shape[0]):
            if max_boxes == n_boxes:
                break
            if scores[i] > min_score_thresh and classes[i] == 1:
                bb.append(tuple(boxes[i].tolist()))
                sc.append(scores[i])
                n_boxes += 1

        if return_annotations_only:
            return bb, sc

        for box, score in zip(bb, sc):
            ymin, xmin, ymax, xmax = box
            draw_bounding_box_on_image_array(
                frame,
                ymin,
                xmin,
                ymax,
                xmax,
                color='green',
                thickness=2,
                display_str_list=[f'person: {int(100 * score)}%'], use_normalized_coordinates=False)

        return frame, bb, sc


def main():
    detector = Detector(download_model=False)

    # noinspection PyShadowingNames
    logger = util.get_logger(__name__, logging.DEBUG)

    from streaming import Stream
    stream = Stream()
    for frame in stream.playing():
        frame, bb, sc = detector.infer(frame, return_annotations_only=False)
        logger.debug(f'{bb}, {sc}')
        cv2.imshow('Frame', frame.astype(np.uint8))
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
