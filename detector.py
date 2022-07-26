"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: March 28, 2022

Purpose: detection module. contains a class which can handle images and run the object detection during inference time
"""
import cv2
import numpy as np
import tensorflow as tf

import util
from automl.efficientdet.tf2 import infer_lib

MODEL = 'efficientdet-lite4'
logger = util.get_logger(__name__)


class Detector:
    """Inference with efficientDet object detector"""

    def __init__(self, *, params, download_model=False, **kwargs):
        """
        init
        :param params: params to pass to object detection model driver encapsulated within this class
        :param download_model: whether to download model if not present as {MODEL}.tgz file
        :param kwargs: to pass to object detection driver
        """
        if download_model:
            # Download checkpoint.
            util.download(MODEL)
            logger.info(f'Using model in {MODEL}')
        self.driver = infer_lib.KerasDriver(MODEL, debug=False, model_name=MODEL, model_params=params, **kwargs)

    def infer(self, frame, max_boxes=200):
        """
        basic inference on a single frame only outputs person detections. this also handles any pre and post processing
        so input frame need not be preprocessed
        :param frame: raw un-preprocessed frame
        :param max_boxes: max number of detections per image to report
        :return: bounding boxes array, scores array for person class
        """
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
            if classes[i] == 1:
                bb.append(tuple(boxes[i].tolist()))
                sc.append(scores[i])
                n_boxes += 1

        logger.debug(f'{bb}, {sc}')
        return bb, sc

    def __call__(self, frame):
        """
        makes this object callable. calls the infer function when this object is called. but instead of returning the
        boxes and scores, it draws the bounding boxes on top of the frame and returns the frame
        :param frame: frame
        :return: frame with detections overlaid
        """
        bb, sc = self.infer(frame)
        bb, sc = util.filter_by_thresh(bb, sc, self.driver.model.config.nms_configs.score_thresh)
        frame = util.draw_boxes(frame, bb, sc)
        return frame


def test():
    """test only"""
    import argparse

    parser = argparse.ArgumentParser(description='detector interface')
    parser.add_argument('--download', dest='download', action='store_true', help='download model')
    parser.add_argument('--no-download', dest='download', action='store_false', help='dont download model (default)')
    parser.add_argument('--filename', dest='filename',
                        help='optional video filename (will use webcam feed if this option is absent)', default=None)
    parser.set_defaults(download=False)

    args = parser.parse_args()
    config_override = {'nms_configs': {'iou_thresh': .5, 'score_thresh': .5},
                       'image_size': 480}
    detector = Detector(params=config_override, download_model=args.download)

    from streaming import Stream
    stream = Stream(args.filename)
    for frame in stream.play():
        frame = detector(frame)

        # show frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Frame', frame.astype(np.uint8))

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    test()
