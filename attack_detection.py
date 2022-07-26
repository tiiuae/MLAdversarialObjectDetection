"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: June 7, 2022

Purpose: a self-supervised method to detect patch attacks on any object detector
"""
import ast
import functools
import os

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
import tfplot
from tifffile import tifffile

import generator
import brightness_matcher
import hparams_config as hparams
import util
from tf2 import postprocess, efficientdet_keras

logger = util.get_logger(__name__)
MODEL = 'efficientdet-lite4'


class PatchAttackDefender(tf.keras.Model):
    """defend against patch attacks"""

    def __init__(self, protege_model: efficientdet_keras.EfficientDetModel, initial_weights=None, eval_patch=None,
                 protege_config_override=None, visualize_freq=200):
        """
        init
        :param protege_model: model to protect.
        :param initial_weights: to initialize training if needed. should be a h5 weights file
        :param eval_patch: target model adversarial patch used only for evaluation phase
        :param protege_config_override: override default hyperparams of protege object detection model. used for setting
        nms threshold and iou threshold during postprocessing
        :param visualize_freq:
        """
        super().__init__(name='Defender_Graph')
        self.protege_model = protege_model
        for layer in protege_model.layers:
            layer.trainable = False
        self.config = self.protege_model.config
        if protege_config_override:
            self.protege_model.config.override(protege_config_override)

        # initialize attention U-net model
        self._antipatch = generator.define_model(self.config.image_size, generator.PatchNeutralizer)
        if initial_weights is not None:
            self._antipatch.load_weights(initial_weights)

        patch = tifffile.imread(os.path.join(eval_patch, 'patch.tiff'))
        with open(os.path.join(eval_patch, 'scale.txt')) as f:
            scale = ast.literal_eval(f.read())

        self._masker = Masker(patch, scale, name='Masker')

        self.visualize_freq = tf.constant(visualize_freq, tf.int64)
        self.cur_step = None
        self.tb = None
        self._trainable_variables = self._antipatch.trainable_variables

    def compile(self, *args, **kwargs):
        """
        compile the defender model
        :param args: superclass args
        :param kwargs: superclass kwargs
        """
        super().compile(*args, **kwargs)
        self._antipatch.compile(*args, **kwargs)

    def filter_valid_boxes(self, images, boxes, scores):
        """
        filter bounding boxes by invalid boxes. i.e., filter box size bigger than the image or smaller than 100px area.
        :param images: images
        :param boxes: predicted bounding boxes
        :param scores: predicted scores
        :return: boolean map over selected bounding boxes
        """
        _, h, w, _ = tf.unstack(tf.cast(tf.shape(images), tf.float32))
        boxes_h = boxes[:, :, 2] - boxes[:, :, 0]
        boxes_w = boxes[:, :, 3] - boxes[:, :, 1]
        boxes_area = boxes_h * boxes_w
        cond1 = tf.logical_and(tf.less_equal(boxes_w / w, 1.), tf.less_equal(boxes_h / h, 1.))
        cond2 = tf.logical_and(tf.greater(boxes_area, tf.constant(100.)),
                               tf.greater_equal(scores, self.config.nms_configs.score_thresh))
        return tf.logical_and(cond1, cond2)

    def odet_model(self, images, score_thresh=None):
        """
        object detection pass
        :param images: images
        :param score_thresh: threhold
        :return: predicted boxes and scores
        """
        if score_thresh is not None:
            config = hparams.Config(self.config.as_dict())
            config.nms_configs.score_thresh = score_thresh
            assert config.nms_configs.score_thresh == score_thresh
        else:
            config = self.config

        with tf.name_scope('object_detection'):
            cls_outputs, box_outputs = self.protege_model(images, pre_mode=None, post_mode=None)
            cls_outputs = postprocess.to_list(cls_outputs)
            box_outputs = postprocess.to_list(box_outputs)
            boxes, scores, classes = postprocess.pre_nms(config.as_dict(), cls_outputs, box_outputs)

            # filter non-relevant classes
            person_indices = tf.equal(classes, tf.constant(0))  # taking postprocess.CLASS_OFFSET into account
            scores = tf.ragged.boolean_mask(scores, person_indices)

            boxes = tf.ragged.boolean_mask(boxes, person_indices)
            classes = tf.ragged.boolean_mask(classes, person_indices)

            # do nms
            boxes, scores = self._postprocessing(boxes, scores, classes, score_thresh=score_thresh)

            # filter valid boxes
            valid_boxes = self.filter_valid_boxes(images, boxes, scores)
            boxes = tf.ragged.boolean_mask(boxes, valid_boxes)
            scores = tf.ragged.boolean_mask(scores, valid_boxes)
        return boxes, scores

    def _postprocessing(self, boxes, scores, score_thresh=None):
        """
        postprocessing output
        :param boxes: predicted boxes
        :param scores: predicted scores
        :param score_thresh: threshold
        :return: nms outputs
        """
        classes = tf.zeros_like(scores)  # needed for tf nms call
        if score_thresh is not None:
            config = hparams.Config(self.config.as_dict())
            config.nms_configs.score_thresh = score_thresh
            assert config.nms_configs.score_thresh == score_thresh
        else:
            config = self.config

        with tf.name_scope('post_processing'):
            def single_batch_fn(element):
                """
                process single image in a batch
                :param element: index in a batch
                :return: nms outputs
                """
                return postprocess.nms(config, boxes[element], scores[element], classes[element], True)

            if not isinstance(boxes, tf.RaggedTensor):
                boxes = tf.RaggedTensor.from_tensor(boxes)
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
        :return: if training return gradients else nothing
        """
        # first pass through object detector to get person bounding boxes
        boxes, scores = self.odet_model(images)

        with tf.GradientTape() as tape:
            # add self supervised patches to persons detected in the first pass and create ground truth targets for the
            # defender model
            # if training is not happening then masker class adds adversarial patches for evaluation instead of
            # self-supervised patches
            images, targets = self._masker([boxes, images], training=training)

            if not training:
                # do a second pass though object detector to evaluate deterioration in outputs due to adversarial patch
                boxes, scores = self.odet_model(images, score_thresh=0.)

            # run the defender model and calculate loss with respect to targets
            updates = 2. * self._antipatch(images, training=training)
            flat_targets = tf.reshape(targets, (tf.shape(images)[0], -1))
            flat_updates = tf.reshape(updates, (tf.shape(images)[0], -1))
            loss = tf.reduce_sum(tf.reduce_mean((flat_targets - flat_updates) ** 2., axis=1))

        # report loss for tensorboard
        self.add_metric(loss, name='loss')

        # decide whether to visualise images during this epoch based on self.visualize_freq and call the vis_images
        # function if so
        func = functools.partial(self.vis_images, images, updates, boxes, scores, training)
        tf.cond(tf.equal(tf.math.floormod(self.cur_step, self.visualize_freq), tf.constant(0, tf.int64)),
                func, lambda: None)

        if training:
            # return gradients
            return tape.gradient(loss, self._trainable_variables)

    @staticmethod
    @tfplot.autowrap(figsize=(4, 4))
    def plot_scores(x, y, step, *, ax):
        """
        seaborn violin plot for score distributions when visualized. this function fetches data back to the cpu and
        must only be called rarely during training. used by vis_images
        :param x: object detection scores on clean image during training and on adversarial image during evaluation
        phase
        :param y: object detection scores after defender model action. if training the score distributions should be
        similar after defender action. during evaluation the scores should improve after defender action
        :param step: irrelevant
        :param ax: mpl axis
        """
        def make_df(arr, label):
            """
            make a pandas dataframe to support seaborn plotting
            :param arr: input array (scores)
            :param label: (original or recovered)
            :return: dataframe
            """
            df = pd.DataFrame(arr, columns=['scores'])
            df['label'] = label
            df[''] = ''  # seaborn issue
            return df

        df1 = make_df(x, 'original')
        df2 = make_df(y, 'recovered')
        df = pd.concat([df1, df2], ignore_index=True)
        ax = sns.violinplot(x='', y='scores', hue='label', split=True, data=df, ax=ax)
        ax.legend()

    def vis_images(self, images, updates, labels, scores, training):
        """
        visualize images on tensorboard with bounding boxes and patches during training
        :param images: images
        :param updates: output from the defender model (to be added to images)
        :param labels: first pass boxes if training else boxes after adv. attack
        :param scores: first pass scores if training else scores after adv. attack
        :param training: True if training else False if testing
        """
        _, h, w, _ = tf.unstack(tf.cast(tf.shape(images), tf.float32))

        tr = 'train' if training else 'val'

        def convert_format(box):
            """
            convert ragged tensor to normal tensor and normalize by image dimensions
            :param box: box
            :return: normalized boxes as normal tensor
            """
            ymin, xmin, ymax, xmax = tf.unstack(box.to_tensor(), axis=2)
            return tf.stack([ymin / h, xmin / w, ymax / h, xmax / w], axis=2)

        labels = convert_format(labels)

        # add defender output to input images to neutralize patches
        updated_images = tf.clip_by_value(images + updates, -1., 1.)

        # pass updated images through object detector
        boxes_pred, scores_pred = self.odet_model(updated_images, score_thresh=0.)
        boxes_pred = convert_format(boxes_pred)

        images = tf.image.draw_bounding_boxes(images, labels, tf.constant([[0., 1., 0.]]))

        # draw stuff
        updated_images = tf.image.draw_bounding_boxes(updated_images, boxes_pred, tf.constant([[0., 0., 1.]]))
        images = tf.clip_by_value(images * self.config.stddev_rgb + self.config.mean_rgb, 0., 255.)
        images = tf.cast(images, tf.uint8)
        updated_images = tf.clip_by_value(updated_images * self.config.stddev_rgb + self.config.mean_rgb, 0., 255.)
        updated_images = tf.cast(updated_images, tf.uint8)
        images = tf.reshape(tf.concat([images, updated_images], axis=1), (-1, *updated_images[0].shape))

        # prepare plots
        scores = scores.merge_dims(0, -1)
        scores_pred = scores_pred.merge_dims(0, -1)
        plot = self.plot_scores(scores, scores_pred, self.cur_step)

        # report to tensorboard
        with self.tb._writers[tr].as_default():
            tf.summary.image('Sample', images, step=self.cur_step, max_outputs=tf.shape(images)[0])
            tf.summary.image('Scores', plot[tf.newaxis], step=self.cur_step)

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
        save defender model weights as h5 file after each epoch
        :param dirpath: directory
        :param kwargs: unused
        """
        os.makedirs(dirpath)
        self._antipatch.save_weights(os.path.join(dirpath, 'antipatch.h5'))


class Masker(tf.keras.layers.Layer):
    """add patches to image. during training phase add self-supervised patches. during evaluation add adv. patches"""

    def __init__(self, patch, scale_regressor, *args, min_patch_area=4, **kwargs):
        """
        init
        :param patch: patch
        :param scale_regressor: scale
        :param args: superclass args
        :param min_patch_area: minimum area in px to be patched. patch will not be downscaled to areas below this size
        :param kwargs: superclass kwargs
        """
        super().__init__(*args, trainable=False, **kwargs)
        self._batch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._patch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._boxes = None
        self._patch = patch
        self.min_patch_area = min_patch_area
        self._matcher = brightness_matcher.BrightnessMatcher(name='Brightness_Matcher')
        self._scale = scale_regressor
        self.is_training = False
        self._train_patches = None

    @staticmethod
    def random_print_adjust(patch):
        """
        simulate variations incurred during printing and reimaging the patch, were it a physical patch
        :return: "printed and reimaged" patch
        """
        w = tf.random.normal((1, 1, 3), .5, .1)
        b = tf.random.normal((1, 1, 3), 0., .01)
        return tf.clip_by_value(w * patch + b, -1., 1.)

    def add_patches_to_image(self, image):
        """
        add patches to a single image possibly containing many persons.
        :param image: image
        :return: patched image
        """
        h, w, _ = tf.unstack(tf.cast(tf.shape(image), tf.float32))
        boxes = self._boxes[self._batch_counter]

        # find coordinates to be patched
        patch_boxes = tf.vectorized_map(functools.partial(self.create, image), boxes)
        patch_boxes = tf.reshape(patch_boxes, shape=(-1, 5))
        valid_indices = tf.where(tf.greater(patch_boxes[:, 2] * patch_boxes[:, 3],
                                            tf.constant(self.min_patch_area, tf.float32)))
        patch_boxes = tf.gather_nd(patch_boxes, valid_indices)

        # behaviour changes according to whether we are in the training or evaluation phase
        if not self.is_training:
            patch = self.random_print_adjust(self._patch)
            patch = self._matcher((patch, image))
        else:
            patch = self._train_patches[self._batch_counter]
            patch = self.random_print_adjust(patch)
            patch = self._matcher((patch, image))

        self._patch_counter.assign(tf.constant(0))
        mask = tf.zeros_like(image)

        # add patches on GPU loop
        loop_fn = functools.partial(self.add_patch_to_image, patch_boxes, patch, image)
        image, mask, _ = tf.while_loop(lambda image, mask, j: tf.less(self._patch_counter, tf.shape(boxes)[0]),
                                       loop_fn, [image, mask, self._patch_counter])

        self._batch_counter.assign_add(tf.constant(1))
        return image, mask

    def add_patch_to_image(self, patch_boxes, patch, oimage, image, mask, j):
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

        # brightness and noise transformation logic
        im = tf.image.resize(patch, tf.stack([patch_h, patch_w]), antialias=True)
        im += tf.random.uniform(shape=tf.shape(im), minval=-.1, maxval=.1)
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
        im = tf.where(tf.less(im, -1.), patch_bg, im)
        im = tf.clip_by_value(im, -1., 1.)

        # patch
        image = tf.tensor_scatter_nd_update(image, idx, im)
        patch_bg = oimage[ymin_patch: ymax_patch, xmin_patch: xmax_patch]
        mask = tf.tensor_scatter_nd_update(mask, idx, patch_bg - im)
        self._patch_counter.assign_add(tf.constant(1))
        return [image, mask, self._patch_counter]

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

        # random variation from bounding box center and random variation of scale during training phase
        if self.is_training:
            tolerance = .5
            scale = tf.random.uniform((), minval=.3, maxval=.5)
        else:
            tolerance = 0.
            scale = self._scale

        patch_size = tf.floor(longer_side * scale)
        diag = tf.minimum((2. ** .5) * patch_size, tf.cast(image.shape[1], tf.float32))

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

    def call(self, inputs, training=False):
        """
        called during training by the attacker for each batch
        :param inputs: set of input images
        :return: patched images with self-supervised patches during training or adversarial patch during evaluation
        """
        self._boxes, images = inputs
        self.is_training = training

        # generate patches in self-supervised way by using other images in the same batch as patches
        if training:
            patches = tf.random.shuffle(images[:, :240, :240, :])
            patches = tf.image.random_flip_left_right(patches)
            patches = tf.image.random_flip_up_down(patches)
            patches = tf.image.adjust_contrast(patches, 2.)
            patches = tf.image.adjust_saturation(patches, 2.)
            self._train_patches = patches
        else:
            self._train_patches = None
        self._batch_counter.assign(tf.constant(0))
        return tf.map_fn(self.add_patches_to_image, images,
                         fn_output_signature=(tf.TensorSpec(dtype=tf.float32, shape=images[0].shape),
                                              tf.TensorSpec(dtype=tf.float32, shape=images[0].shape)))
