"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 28, 2022

Purpose: attack the person detector with dynamic patches. All the logic inside this module is executed in a tf Graph
placed entirely on the GPU
"""
import ast
import functools
import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tfplot
from matplotlib import pyplot as plt
from tifffile import tifffile

import brightness_matcher
from tf2 import postprocess, efficientdet_keras


class PatchAttacker(tf.keras.Model):
    """attack with malicious patches"""

    def __init__(self, model: efficientdet_keras.EfficientDetModel, initial_patch=None,
                 config_override=None, visualize_freq=200):
        """
        init
        :param model: victim model
        :param initial_patch: initial patch image in tiff format or None to initialize with a random patch
        :param config_override: override default hyperparams of victim model. used for setting nms threshold and
        iou threshold during postprocessing stage
        :param visualize_freq: write visualization summary at every nth epoch denoted by this number
        """
        super().__init__(name='Attacker_Graph')
        self.model = model
        self.config = self.model.config
        if config_override:
            self.model.config.override(config_override)
        if initial_patch is None:
            patch_img = np.random.uniform(-1., 1., size=(640, 640, 3))
            scale = .4
        else:
            patch_img = tifffile.imread(os.path.join(initial_patch, 'patch.tiff'))
            with open(os.path.join(initial_patch, 'scale.txt')) as f:
                scale = ast.literal_eval(f.read())

        # these two are the only variables to be updated during training
        self._patch = tf.Variable(patch_img, trainable=True, name='patch', dtype=tf.float32,
                                  constraint=lambda x: tf.clip_by_value(x, -1., 1.))
        self._scale_regressor = tf.Variable(scale, trainable=True, name='scale', dtype=tf.float32,
                                            constraint=lambda x: tf.clip_by_value(x, 0., 1.))

        self.visualize_freq = tf.constant(visualize_freq, tf.int64)

        # this object is responsible for patch transformations and application onto an image
        self._patcher = Patcher(self._patch, self._scale_regressor, name='Patcher')

        self.cur_step = None
        self.tb = None
        self._trainable_variables = [self._scale_regressor, self._patch]

        # for attack success rate calculation
        self.bins = np.arange(self.config.nms_configs.score_thresh, .805, .01, dtype='float32')
        self.asr = [tf.Variable(0., dtype=tf.float32, trainable=False) for _ in self.bins]

    def filter_valid_boxes(self, images, boxes, scores, thresh=True):
        """
        filter bounding boxes by invalid boxes. i.e., filter box size bigger than the image or smaller than 100px area.
        optionally also by threshold
        :param images: images
        :param boxes: predicted bounding boxes
        :param scores: predicted scores
        :param thresh: if True then also filter by threshold defined in config
        :return: boolean map over selected bounding boxes
        """
        _, h, w, _ = tf.unstack(tf.cast(tf.shape(images), tf.float32))
        boxes_h = boxes[:, :, 2] - boxes[:, :, 0]
        boxes_w = boxes[:, :, 3] - boxes[:, :, 1]
        boxes_area = boxes_h * boxes_w
        cond1 = tf.logical_and(tf.less_equal(boxes_w / w, 1.), tf.less_equal(boxes_h / h, 1.))
        if thresh:
            cond2 = tf.logical_and(tf.greater(boxes_area, tf.constant(100.)),
                                   tf.greater_equal(scores, self.config.nms_configs.score_thresh))
        else:
            cond2 = tf.greater(boxes_area, tf.constant(100.))
        return tf.logical_and(cond1, cond2)

    def first_pass(self, images):
        """
        clean pass (without patches) through target victim model to obtain predictions.
        :param images: input
        :return: predicted boxes and scores for the person class only
        """
        # note that below pre and post processing is disabled in google's implementation since we're doing it ourselves
        cls_outputs, box_outputs = self.model(images, pre_mode=None, post_mode=None)

        with tf.name_scope('first_pass'):
            cls_outputs = postprocess.to_list(cls_outputs)
            box_outputs = postprocess.to_list(box_outputs)
            boxes, scores, classes = postprocess.pre_nms(self.config.as_dict(), cls_outputs, box_outputs)

            # filter irrelevant classes except person
            person_indices = tf.equal(classes, tf.constant(0))  # taking postprocess.CLASS_OFFSET into account
            scores = tf.ragged.boolean_mask(scores, person_indices)
            boxes = tf.ragged.boolean_mask(boxes, person_indices)

            # further filter invalid boxes
            valid_boxes = self.filter_valid_boxes(images, boxes, scores)
            boxes = tf.ragged.boolean_mask(boxes, valid_boxes)
            scores = tf.ragged.boolean_mask(scores, valid_boxes)
            boxes, scores = self._postprocessing(boxes, scores)

        return boxes, scores

    def second_pass(self, images):
        """
        second pass through target model after addition of patches
        :param images: input images with adv. patches attached to detections in the first pass
        :return: predicted boxes and scores for the person class only
        """
        # note that below pre and post processing is disabled in google's implementation since we're doing it ourselves
        cls_outputs, box_outputs = self.model(images, pre_mode=None, post_mode=None)

        with tf.name_scope('attack_pass'):
            cls_outputs = postprocess.to_list(cls_outputs)
            box_outputs = postprocess.to_list(box_outputs)
            boxes, scores, classes = postprocess.pre_nms(self.config.as_dict(), cls_outputs, box_outputs)

            # filter irrelevant classes except person
            person_indices = tf.equal(classes, tf.constant(0))  # taking postprocess.CLASS_OFFSET into account
            scores = tf.ragged.boolean_mask(scores, person_indices)
            boxes = tf.ragged.boolean_mask(boxes, person_indices)

            # further filter invalid boxes
            valid_boxes = self.filter_valid_boxes(images, boxes, scores, thresh=False)
            scores = tf.ragged.boolean_mask(scores, valid_boxes)
            boxes = tf.ragged.boolean_mask(boxes, valid_boxes)
        return boxes, scores

    def _postprocessing(self, boxes, scores):
        """
        postprocessing output
        :param boxes: predicted boxes
        :param scores: predicted scores
        :return: non maximum suppressed output boxes and scores
        """
        classes = tf.zeros_like(scores)  # needed for tf nms call
        with tf.name_scope('post_processing'):
            def single_batch_fn(element):
                """
                process single image in a batch
                :param element: index in a batch
                :return: nms outputs
                """
                return postprocess.nms(self.config, boxes[element], scores[element], classes[element], True)

            nms_boxes, nms_scores, nms_classes, nms_valid_len = tf.map_fn(single_batch_fn, tf.range(boxes.nrows()),
                                                                          fn_output_signature=(
                                                                              tf.TensorSpec((None, 4),
                                                                                            dtype=tf.float32),
                                                                              tf.TensorSpec((None,), dtype=tf.float32),
                                                                              tf.TensorSpec((None,), dtype=tf.float32),
                                                                              tf.TensorSpec((), dtype=tf.int32)))
            nms_boxes = postprocess.clip_boxes(nms_boxes, self.config.image_size)
            nms_boxes = tf.RaggedTensor.from_tensor(nms_boxes, lengths=nms_valid_len)
            nms_scores = tf.RaggedTensor.from_tensor(nms_scores, lengths=nms_valid_len)
        return nms_boxes, nms_scores

    def call(self, images, *, training=True):
        """
        called on each batch
        :param images: input images
        :param training: training or not
        :return: predicted boxes and scores for the person class only if not training else return gradients
        """
        # first pass clean to get bounding boxes of persons
        boxes, scores = self.first_pass(images)

        with tf.GradientTape() as tape:
            # add patch to detected persons with first pass
            images = self._patcher([boxes, images])

            # make second pass with adv. altered images
            boxes_pred, scores_pred = self.second_pass(images)

            # calculate loss
            max_scores = tf.maximum(tf.reduce_max(scores_pred, axis=1), 0.)
            scale_losses = (max_scores - self._scale_regressor) ** 2.
            tv_loss = tf.image.total_variation(self._patch.value())
            loss = tf.reduce_sum(max_scores ** 2. + scale_losses) + 1e-5 * tv_loss

        # report metrics for tensorboard
        self.add_metric(loss, name='loss')
        self.add_metric(self._scale_regressor.value(), name='scale')
        self.add_metric(tf.reduce_sum(scale_losses), name='scale_loss')
        self.add_metric(tv_loss, name='tv_loss')
        self.add_metric(tf.reduce_mean(max_scores), name='mean_max_score')
        self.add_metric(tf.math.reduce_std(max_scores), name='std_max_score')

        # attack success rate calculation
        boxes_pred, scores_pred = self._postprocessing(boxes_pred, scores_pred)
        asr = self.calc_asr(scores, scores_pred)
        self.add_metric(asr, name='asr')
        self.add_metric(asr / self._scale_regressor.value(), name='asr_to_scale')

        # decide whether to visualise images during this epoch based on self.visualize_freq and call the vis_images
        # function if so
        vis_func = functools.partial(self.vis_images, images, boxes, scores, boxes_pred, scores_pred, training)
        tf.cond(tf.equal(tf.math.floormod(self.cur_step, self.visualize_freq), tf.constant(0, tf.int64)),
                vis_func, lambda: None)

        if training:
            # return gradients
            return tape.gradient(loss, self._trainable_variables)

        return boxes_pred, scores_pred

    @staticmethod
    @tfplot.autowrap(figsize=(4, 4))
    def plot_asr(x: np.ndarray, y: np.ndarray, step, *, ax, color='blue'):
        """
        matplotlib plotting for attack success rate when visualized. this function fetches data back to the cpu and must
        only be called rarely during training. used by vis_images
        :param x: score thresholds
        :param y: attack success rate at those thresholds
        :param step: irrelevant
        :param ax: mpl axis
        :param color: color
        """
        ax.plot(x, y, color=color)
        ax.set_ylim(0., 1.)
        ax.set_xlabel('score_thresh')
        ax.set_ylabel('attack_success_rate')

    def calc_asr(self, scores, scores_pred, *, score_thresh=.5):
        """
        calculate attack success rate at a given score threshold
        :param scores: first pass scores
        :param scores_pred: second pass scores
        :param score_thresh: threshold
        :return: attack success rate
        """
        filt = tf.greater_equal(scores, tf.constant(score_thresh))
        scores_filt = tf.ragged.boolean_mask(scores, filt)

        filt = tf.greater_equal(scores_pred, tf.constant(score_thresh))
        scores_pred_filt = tf.ragged.boolean_mask(scores_pred, filt)
        return 1. - tf.cast(tf.size(scores_pred_filt.flat_values),
                            tf.float32) / (tf.cast(tf.size(scores_filt.flat_values), tf.float32) +
                                           tf.keras.backend.epsilon())

    def vis_images(self, images, labels, scores, boxes_pred, scores_pred, training):
        """
        visualize images on tensorboard with bounding boxes and patches during training
        :param images: images
        :param labels: first pass boxes
        :param scores: first pass scores
        :param boxes_pred: second pass boxes
        :param scores_pred: second pass scores
        :param training: True if training else False if testing
        """
        _, h, w, _ = tf.unstack(tf.cast(tf.shape(images), tf.float32))
        tr = 'train' if training else 'val'

        # report patch to tensorboard
        with self.tb._writers[tr].as_default():
            if training:
                patch = tf.clip_by_value(self._patch * self.config.stddev_rgb + self.config.mean_rgb, 0., 255.)
                patch = tf.cast(patch, tf.uint8)
                tf.summary.image('Patch', patch[tf.newaxis], step=self.cur_step)

        # calculate attack success rate and report to tensorboard
        for i, score in enumerate(self.bins):
            self.asr[i].assign(self.calc_asr(scores, scores_pred, score_thresh=score))
        plot = self.plot_asr(self.bins, self.asr, self.cur_step)

        with self.tb._writers[tr].as_default():
            tf.summary.image('ASR', plot[tf.newaxis], step=self.cur_step)

        def convert_format(box):
            """
            convert ragged tensor to normal tensor and normalize by image dimensions
            :param box: box
            :return: normalized boxes as normal tensor
            """
            ymin, xmin, ymax, xmax = tf.unstack(box.to_tensor(), axis=2)
            return tf.stack([ymin / h, xmin / w, ymax / h, xmax / w], axis=2)

        labels = convert_format(labels)
        boxes_pred = convert_format(boxes_pred)

        # draw stuff
        images = tf.image.draw_bounding_boxes(images, labels, tf.constant([[0., 1., 0.]]))
        images = tf.image.draw_bounding_boxes(images, boxes_pred, tf.constant([[0., 0., 1.]]))
        images = tf.clip_by_value(images * self.config.stddev_rgb + self.config.mean_rgb, 0., 255.)
        images = tf.cast(images, tf.uint8)

        # report to tensorboard
        with self.tb._writers[tr].as_default():
            tf.summary.image('Sample', images, step=self.cur_step, max_outputs=tf.shape(images)[0])

    def train_step(self, inputs):
        """
        called for each batch during training
        :param inputs: batch
        :return: metrics as a dict
        """
        self.cur_step = self.tb._train_step
        grads = self(inputs)
        self.optimizer.apply_gradients([*zip(grads, self._trainable_variables)])
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        """
        called for each batch during validation phase after each epoch
        :param inputs: batch
        :return: metrics as a dict
        """
        self.cur_step = self.tb._val_step
        self(inputs, training=False)
        return {m.name: m.result() for m in self.metrics}

    def save_weights(self, dirpath, **kwargs):
        """
        save patch and current scale to disk after each epoch
        :param dirpath: directory
        :param kwargs: unused
        """
        os.makedirs(dirpath)
        with open(os.path.join(dirpath, 'scale.txt'), 'w') as f:
            f.write(str(self._scale_regressor.numpy()))

        patch = tf.clip_by_value(self._patch * self.config.stddev_rgb + self.config.mean_rgb, 0., 255.)
        patch = tf.cast(patch, tf.uint8).numpy()
        plt.imsave(os.path.join(dirpath, 'patch.png'), patch)
        tifffile.imwrite(os.path.join(dirpath, 'patch.tiff'), self._patch.numpy())


class Patcher(tf.keras.layers.Layer):
    """apply patch to persons in an image"""

    def __init__(self, patch: tf.Variable, scale_regressor: tf.Variable, *args, min_patch_area=4, **kwargs):
        """
        init
        :param patch: patch variable
        :param scale_regressor: scale variable
        :param args: superclass args
        :param min_patch_area: minimum area in px to be patched. patch will not be downscaled to areas below this size
        :param kwargs: superclass kwargs
        """
        super().__init__(*args, trainable=False, **kwargs)
        self._patch = patch
        self._batch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._patch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._boxes = None
        self.min_patch_area = min_patch_area
        self._matcher = brightness_matcher.BrightnessMatcher(name='Brightness_Matcher')
        self._scale = scale_regressor

    def random_print_adjust(self):
        """
        simulate variations incurred during printing and reimaging the patch, were it a physical patch
        :return: "printed and reimaged" patch
        """
        w = tf.random.normal((1, 1, 3), .5, .1)
        b = tf.random.normal((1, 1, 3), 0., .01)
        return tf.clip_by_value(w * self._patch + b, -1., 1.)

    def add_patches_to_image(self, image):
        """
        add patches to a single image possibly containing many persons.
        :param image: image
        :return: patched image
        """
        h, w, _ = tf.unstack(tf.cast(tf.shape(image), tf.float32))
        boxes = self._boxes[self._batch_counter]

        # printer random color variation transformation
        patch = self.random_print_adjust()

        # scene brightness match transformation
        patch = self._matcher((patch, image))

        # find coordinates to be patched within the image
        patch_boxes = tf.vectorized_map(functools.partial(self.create, image), boxes)
        patch_boxes = tf.reshape(patch_boxes, shape=(-1, 5))
        valid_indices = tf.where(tf.greater(patch_boxes[:, 2] * patch_boxes[:, 3],
                                            tf.constant(self.min_patch_area, tf.float32)))
        patch_boxes = tf.gather_nd(patch_boxes, valid_indices)

        # run in GPU loop
        self._patch_counter.assign(tf.constant(0))
        loop_fn = functools.partial(self.add_patch_to_image, patch_boxes, patch)
        image, _ = tf.while_loop(lambda image, j: tf.less(self._patch_counter, tf.shape(patch_boxes)[0]),
                                 loop_fn, [image, self._patch_counter])

        self._batch_counter.assign_add(tf.constant(1))
        return image

    def add_patch_to_image(self, patch_boxes, patch, image, j):
        """
        add single patch to an image. called by the previous function in a GPU hosted while loop. this method also
        applies rotational transformation +- 20 degrees before applying the patch and adds a small random noise to
        simulate camera sensor noise in "reimaging" the "physical" patch. it also varies the brightness of the patch in
        small random amount to simulate local lighting within the image
        :param patch_boxes: set of patches for this image
        :param patch: patch tensor to be applied after print and brightness match variations
        :param image: image
        :param j: index variable for signature compatibility. unsused. actual indexing into patch_boxes is done by a
        class field self._patch_counter
        :return: image and patch_index as loop variables
        """
        ymin_patch, xmin_patch, patch_h, patch_w, diag = tf.unstack(tf.cast(patch_boxes[self._patch_counter], tf.int32))
        ymax_patch = ymin_patch + diag
        xmax_patch = xmin_patch + diag
        idx = tf.stack(tf.meshgrid(tf.range(ymin_patch, ymax_patch), tf.range(xmin_patch, xmax_patch), indexing='ij'),
                       axis=-1)

        # resize, add noise and random brightness tranformations
        im = tf.image.resize(patch, tf.stack([patch_h, patch_w]), antialias=True)
        im += tf.random.uniform(shape=tf.shape(im), minval=-.01, maxval=.01)
        im = tf.image.random_brightness(im, .3)
        im = tf.clip_by_value(im, -1., 1.)

        # rotation logic
        offset = (diag - patch_h) / 2
        top = left = tf.cast(tf.math.floor(offset), tf.int32)
        bottom = right = tf.cast(tf.math.ceil(offset), tf.int32)
        pads = tf.reshape(tf.stack([top, bottom, left, right, 0, 0]), (-1, 2))
        im = tf.pad(im, pads, constant_values=-2)
        angle = tf.random.uniform(shape=(), minval=-20. * np.pi / 180., maxval=20. * np.pi / 180.)
        im = tfa.image.rotate(im, angle, interpolation='bilinear', fill_value=-2.)
        patch_bg = image[ymin_patch: ymax_patch, xmin_patch: xmax_patch]
        # tf.print(tf.shape(im), tf.shape(patch_bg))
        im = tf.where(tf.less(im, -1.), patch_bg, im)
        im = tf.clip_by_value(im, -1., 1.)

        # patch
        image = tf.tensor_scatter_nd_update(image, idx, im)
        self._patch_counter.assign_add(tf.constant(1))
        return [image, self._patch_counter]

    def create(self, image, box):
        """
        creates the coordinates of the area to be patched based on the person bounding box taking into account the
        rotation of the patch as well. called in loop on GPU for all persons in an image
        :param image: image
        :param box: person bounding box in the image
        :return: patch coordinates and the diagonal length which accounts for the change in area during rotation of
        the patch
        """
        ymin, xmin, ymax, xmax = tf.unstack(box, 4)

        h = ymax - ymin
        w = xmax - xmin

        longer_side = tf.maximum(h, w)

        # shift the patch randomly from the person bounding box center within this amount
        tolerance = .2

        # determine the patch size based on current scalar variable (under training)
        patch_size = tf.floor(longer_side * self._scale)

        diag = tf.minimum((2. ** .5) * patch_size, tf.cast(image.shape[1], tf.float32))
        # tf.print(patch_size)

        orig_y = ymin + h / 2. + tf.random.uniform((), minval=-tolerance * h / 2., maxval=tolerance * h / 2.)
        orig_x = xmin + w / 2. + tf.random.uniform((), minval=-tolerance * w / 2., maxval=tolerance * w / 2.)

        patch_w = patch_size
        patch_h = patch_size

        # ensure coordinates are valid, i.e., lie within the image and not outside
        ymin_patch = tf.maximum(orig_y - diag / 2., 0.)
        xmin_patch = tf.maximum(orig_x - diag / 2., 0.)
        shape = tf.cast(tf.shape(image), tf.float32)
        ymin_patch = tf.cond(tf.greater(ymin_patch + diag, shape[0]),
                             lambda: shape[0] - diag, lambda: ymin_patch)
        xmin_patch = tf.cond(tf.greater(xmin_patch + diag, shape[1]),
                             lambda: shape[1] - diag, lambda: xmin_patch)

        return tf.stack([ymin_patch, xmin_patch, patch_h, patch_w, diag])

    def call(self, inputs):
        """
        called during training by the attacker for each batch
        :param inputs: set of input images
        :return: patched images with current adversarial patch
        """
        self._boxes, images = inputs
        self._batch_counter.assign(tf.constant(0))
        return tf.map_fn(self.add_patches_to_image, images)
