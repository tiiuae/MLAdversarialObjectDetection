"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 28, 2022

Purpose: attack the person detector with dynamic patches
"""
import ast
import functools
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tfplot
from tifffile import tifffile

import custom_callbacks
import histogram_matcher
import metrics
import train_data_generator
import util
from tf2 import postprocess, efficientdet_keras

logger = util.get_logger(__name__)
MODEL = 'efficientdet-lite4'


class DynamicPatchAttacker(tf.keras.Model):
    """attack with malicious patches"""

    def __init__(self, model: efficientdet_keras.EfficientDetModel, initial_weights=None,
                 min_patch_area=4, config_override=None, visualize_freq=200):
        super().__init__(name='Attacker_Graph')
        self.model = model
        self.config = self.model.config
        if config_override:
            self.model.config.override(config_override)
        if initial_weights is None:
            patch_img = np.random.uniform(-1., 1., size=(512, 512, 3))
            scale = .5
        else:
            patch_img = tifffile.imread(os.path.join(initial_weights, 'patch.tiff'))
            with open(os.path.join(initial_weights, 'scale.txt')) as f:
                scale = ast.literal_eval(f.read())

        self._patch = tf.Variable(patch_img, trainable=True, name='patch', dtype=tf.float32,
                                  constraint=lambda x: tf.clip_by_value(x, -1., 1.))
        self._scale_regressor = tf.Variable(scale, trainable=True, name='scale', dtype=tf.float32,
                                            constraint=lambda x: tf.clip_by_value(x, 0., 1.))
        self.visualize_freq = tf.constant(visualize_freq, tf.int64)
        self._patcher = Patcher(self._patch, self._scale_regressor, min_patch_area=min_patch_area, name='Patcher')
        self.cur_step = None
        self.tb = None
        self._trainable_variables = [self._scale_regressor, self._patch]

        iou = self.config.nms_configs.iou_thresh
        self._metric = metrics.AttackSuccessRate(iou_thresh=iou)
        self.bins = np.arange(self.config.nms_configs.score_thresh, .805, .01, dtype='float32')
        self.asr = [tf.Variable(0., dtype=tf.float32, trainable=False) for _ in self.bins]

    def filter_valid_boxes(self, boxes):
        boxes_h = boxes[:, :, 2] - boxes[:, :, 0]
        boxes_w = boxes[:, :, 3] - boxes[:, :, 1]
        boxes_area = boxes_h * boxes_w
        return tf.greater(boxes_area, tf.constant(100.))

    def first_pass(self, images):
        with tf.name_scope('first_pass'):
            cls_outputs, box_outputs = self.model(images, pre_mode=None, post_mode=None)
            cls_outputs = postprocess.to_list(cls_outputs)
            box_outputs = postprocess.to_list(box_outputs)
            boxes, scores, classes = postprocess.pre_nms(self.config.as_dict(), cls_outputs, box_outputs)
            person_indices = tf.equal(classes, tf.constant(0))  # taking postprocess.CLASS_OFFSET into account
            scores = tf.ragged.boolean_mask(scores, person_indices)

            boxes = tf.ragged.boolean_mask(boxes, person_indices)
            classes = tf.ragged.boolean_mask(classes, person_indices)

            boxes, scores = self._postprocessing(boxes, scores, classes)

            valid_boxes = self.filter_valid_boxes(boxes)
            boxes = tf.ragged.boolean_mask(boxes, valid_boxes)
            scores = tf.ragged.boolean_mask(scores, valid_boxes)
        return boxes, scores

    def second_pass(self, images, training=True):
        with tf.name_scope('attack_pass'):
            cls_outputs, box_outputs = self.model(images, pre_mode=None, post_mode=None, training=training)
            cls_outputs = postprocess.to_list(cls_outputs)
            box_outputs = postprocess.to_list(box_outputs)
            boxes, scores, classes = postprocess.pre_nms(self.config.as_dict(), cls_outputs, box_outputs)
            person_indices = tf.equal(classes, tf.constant(0))  # taking postprocess.CLASS_OFFSET into account
            scores = tf.ragged.boolean_mask(scores, person_indices)

            boxes = tf.ragged.boolean_mask(boxes, person_indices)
            classes = tf.ragged.boolean_mask(classes, person_indices)

            _, h, w, _ = tf.unstack(tf.cast(tf.shape(images), tf.float32))
            boxes_h = boxes[:, :, 2] - boxes[:, :, 0]
            boxes_w = boxes[:, :, 3] - boxes[:, :, 1]

            valid_boxes = tf.logical_and(tf.less_equal(boxes_w / w, 1.), tf.less_equal(boxes_h / h, 1.))
            scores = tf.ragged.boolean_mask(scores, valid_boxes)
            boxes = tf.ragged.boolean_mask(boxes, valid_boxes)
            classes = tf.ragged.boolean_mask(classes, valid_boxes)
        return boxes, scores, classes

    def _postprocessing(self, boxes, scores, classes):
        with tf.name_scope('post_processing'):
            def single_batch_fn(element):
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
        boxes, scores = self.first_pass(images)

        with tf.GradientTape() as tape:
            images = self._patcher([boxes, images])
            boxes_pred, scores_pred, classes = self.second_pass(images, training=training)
            sc_losses = tf.reduce_sum(scores_pred ** 2., axis=1)
            scale_losses = tf.reduce_max((scores_pred - self._scale_regressor) ** 2., axis=1)
            loss = tf.reduce_sum(sc_losses + scale_losses) + 1. - util.iou_loss(boxes_pred, boxes)

        self.add_metric(loss, name='loss')
        self.add_metric(self._scale_regressor.value(), name='scale')
        self.add_metric(tf.reduce_sum(scale_losses), name='scale_loss')
        self.add_metric(tf.reduce_mean(tf.reduce_max(scores_pred, axis=1)), name='mean_max_score')

        boxes_pred, scores_pred = self._postprocessing(boxes_pred, scores_pred, classes)
        self.add_metric(self.calc_asr(scores, scores_pred, boxes, boxes_pred), name='asr')

        func = functools.partial(self.vis_images, images, boxes, scores, boxes_pred, scores_pred, training)
        tf.cond(tf.equal(tf.math.floormod(self.cur_step, self.visualize_freq), tf.constant(0, tf.int64)),
                func, lambda: None)

        if training:
            return tape.gradient(loss, self._trainable_variables)

        return boxes_pred, scores_pred

    @staticmethod
    @tfplot.autowrap(figsize=(4, 4))
    def plot_asr(x: np.ndarray, y: np.ndarray, step, *, ax, color='blue'):
        ax.plot(x, y, color=color)
        ax.set_ylim(0., 1.)
        ax.set_xlabel('score_thresh')
        ax.set_ylabel('attack_success_rate')

    def calc_asr(self, scores, scores_pred, boxes, boxes_pred, *, score_thresh=.5):
        filt = tf.greater_equal(scores, tf.constant(score_thresh))
        labels_filt = tf.ragged.boolean_mask(boxes, filt)

        filt = tf.greater_equal(scores_pred, tf.constant(score_thresh))
        boxes_pred_filt = tf.ragged.boolean_mask(boxes_pred, filt)
        return self._metric(boxes_pred_filt, labels_filt)

    def vis_images(self, images, labels, scores, boxes_pred, scores_pred, training):
        _, h, w, _ = tf.unstack(tf.cast(tf.shape(images), tf.float32))

        tr = 'train' if training else 'val'
        with self.tb._writers[tr].as_default():
            if training:
                patch = tf.clip_by_value(self._patch * self.config.stddev_rgb + self.config.mean_rgb, 0., 255.)
                patch = tf.cast(patch, tf.uint8)
                tf.summary.image('Patch', patch[tf.newaxis], step=self.cur_step)

        if training:
            for i, score in enumerate(self.bins):
                self.asr[i].assign(.5 * self.asr[i].value() +
                                   .5 * self.calc_asr(scores, scores_pred, labels, boxes_pred, score_thresh=score))
            plot = self.plot_asr(self.bins, self.asr, self.cur_step)

            with self.tb._writers[tr].as_default():
                tf.summary.image('ASR', plot[tf.newaxis], step=self.cur_step)

        def convert_format(box):
            ymin, xmin, ymax, xmax = tf.unstack(box.to_tensor(), axis=2)
            return tf.stack([ymin / h, xmin / w, ymax / h, xmax / w], axis=2)

        labels = convert_format(labels)
        boxes_pred = convert_format(boxes_pred)
        images = tf.image.draw_bounding_boxes(images, labels, tf.constant([[0., 1., 0.]]))
        images = tf.image.draw_bounding_boxes(images, boxes_pred, tf.constant([[0., 0., 1.]]))
        images = tf.clip_by_value(images * self.config.stddev_rgb + self.config.mean_rgb, 0., 255.)
        images = tf.cast(images, tf.uint8)

        with self.tb._writers[tr].as_default():
            tf.summary.image('Sample', images, step=self.cur_step, max_outputs=tf.shape(images)[0])

    def train_step(self, inputs):
        self.cur_step = self.tb._train_step
        grads = self(inputs)
        self.optimizer.apply_gradients([*zip(grads, self._trainable_variables)])
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        self.cur_step = self.tb._val_step
        self(inputs, training=False)
        return {m.name: m.result() for m in self.metrics}

    def save_weights(self, dirpath, **kwargs):
        os.makedirs(dirpath)
        with open(os.path.join(dirpath, 'scale.txt'), 'w') as f:
            f.write(str(self._scale_regressor.numpy()))

        patch = tf.clip_by_value(self._patch * self.config.stddev_rgb + self.config.mean_rgb, 0., 255.)
        patch = tf.cast(patch, tf.uint8).numpy()
        plt.imsave(os.path.join(dirpath, 'patch.png'), patch)
        tifffile.imwrite(os.path.join(dirpath, 'patch.tiff'), self._patch.numpy())


class Patcher(tf.keras.layers.Layer):
    def __init__(self, patch: tf.keras.Model, scale_regressor, *args, min_patch_area=60, **kwargs):
        super().__init__(*args, trainable=False, **kwargs)
        self._patch = patch
        self._batch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._patch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._boxes = None
        self.min_patch_area = min_patch_area
        self._matcher = histogram_matcher.BrightnessMatcher(name='Brightness_Matcher')
        self._scale = scale_regressor

    def random_print_adjust(self):
        w = tf.random.normal((1, 1, 3), .8, .01)
        b = tf.random.normal((1, 1, 3), -.2, .01)
        return tf.clip_by_value(w * self._patch + b, -1., 1.)

    def add_patches_to_image(self, image):
        h, w, _ = tf.unstack(tf.cast(tf.shape(image), tf.float32))
        boxes = self._boxes[self._batch_counter]
        # printer random color variation
        patch = self.random_print_adjust()
        patch = self._matcher((patch, image))

        patch_boxes = tf.vectorized_map(functools.partial(self.create, image), boxes)
        patch_boxes = tf.reshape(patch_boxes, shape=(-1, 4))
        valid_indices = tf.where(tf.greater(patch_boxes[:, 2] * patch_boxes[:, 3],
                                            tf.constant(self.min_patch_area, tf.float32)))
        patch_boxes = tf.gather_nd(patch_boxes, valid_indices)

        self._patch_counter.assign(tf.constant(0))
        loop_fn = functools.partial(self.add_patch_to_image, patch_boxes, patch)
        image, _ = tf.while_loop(lambda image, j: tf.less(self._patch_counter, tf.shape(patch_boxes)[0]),
                                 loop_fn, [image, self._patch_counter])

        self._batch_counter.assign_add(tf.constant(1))
        return image

    def add_patch_to_image(self, patch_boxes, patch, image, j):
        ymin_patch, xmin_patch, patch_h, patch_w = tf.unstack(tf.cast(patch_boxes[self._patch_counter], tf.int32))
        ymax = ymin_patch + patch_h
        xmax = xmin_patch + patch_w
        idx = tf.stack(tf.meshgrid(tf.range(ymin_patch, ymax), tf.range(xmin_patch, xmax), indexing='ij'), axis=-1)

        im = tf.image.resize(patch, tf.stack([patch_h, patch_w]), antialias=True)
        im += tf.random.uniform(shape=tf.shape(im), minval=-.01, maxval=.01)
        im = tf.image.random_brightness(im, .3)
        im = tf.clip_by_value(im, -1., 1.)

        image = tf.tensor_scatter_nd_update(image, idx, im)
        self._patch_counter.assign_add(tf.constant(1))
        return [image, self._patch_counter]

    def create(self, image, item):
        ymin, xmin, ymax, xmax = tf.unstack(item, 4)

        h = ymax - ymin
        w = xmax - xmin

        area = h * w
        tolerance = .2

        patch_size = tf.floor(tf.sqrt(area * self._scale))
        orig_y = ymin + h / 2. + tf.random.uniform((), minval=-tolerance * h / 2., maxval= tolerance * h / 2.)
        orig_x = xmin + w / 2. + tf.random.uniform((), minval=-tolerance * w / 2., maxval= tolerance * w / 2.)

        patch_w = patch_size
        patch_h = patch_size

        ymin_patch = tf.maximum(orig_y - patch_h / 2., 0.)
        xmin_patch = tf.maximum(orig_x - patch_w / 2., 0.)

        shape = tf.cast(tf.shape(image), tf.float32)
        ymin_patch = tf.cond(tf.greater(ymin_patch + patch_h, shape[0]),
                             lambda: shape[0] - patch_h, lambda: ymin_patch)
        xmin_patch = tf.cond(tf.greater(xmin_patch + patch_w, shape[1]),
                             lambda: shape[1] - patch_w, lambda: xmin_patch)

        return tf.stack([ymin_patch, xmin_patch, patch_h, patch_w])

    def call(self, inputs):
        self._boxes, images = inputs
        self._batch_counter.assign(tf.constant(0))
        return tf.map_fn(self.add_patches_to_image, images)


def main(download_model=False):
    from PIL import Image
    # noinspection PyShadowingNames
    logger = util.get_logger(__name__, logging.DEBUG)

    log_dir = util.ensure_empty_dir('log_dir')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # tf.config.run_functions_eagerly(True)

    victim_model = get_victim_model(download_model)
    config_override = {'nms_configs': {'iou_thresh': .5, 'score_thresh': .5}, 'image_size': 300}
    model = DynamicPatchAttacker(victim_model, config_override=config_override, visualize_freq=1)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3), run_eagerly=False)

    datasets: dict = train_data_generator.partition(model.config, 'downloaded_images', 'labels',
                                                    batch_size=1, shuffle=True)

    train_ds = datasets['train']['dataset']
    val_ds = datasets['val']['dataset']
    train_len = datasets['train']['length']
    val_len = datasets['val']['length']
    tb_callback = custom_callbacks.TensorboardCallback(log_dir, write_graph=True, write_steps_per_second=True)
    model.tb = tb_callback

    save_dir = util.ensure_empty_dir('save_dir')
    save_file = 'patch_{epoch:02d}_{val_loss:.4f}.h5'
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=100,
                        steps_per_epoch=20,  # train_len,
                        validation_steps=20,  # val_len,
                        callbacks=[tb_callback,
                                   tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, save_file),
                                                                      monitor='val_loss',
                                                                      verbose=1,
                                                                      save_best_only=False,
                                                                      save_weights_only=True,
                                                                      mode='auto',
                                                                      save_freq='epoch',
                                                                      options=None,
                                                                      initial_value_threshold=None
                                                                      )
                                   ])
    patch = Image.fromarray(model.get_patch().astype('uint8'))
    patch.show('patch')


if __name__ == '__main__':
    main()
