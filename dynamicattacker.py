"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 28, 2022

Purpose: attack the person detector with dynamic patches
"""
import functools
import logging
import os

import tensorflow as tf

import custom_callbacks
import generator
import histogram_matcher
import metrics
import train_data_generator
import util
from tf2 import postprocess, efficientdet_keras, infer_lib

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
        self._patch = generator.define_generator(self.config.image_size)
        if initial_weights is not None:
            self._patch.load_weights(initial_weights)
        self.visualize_freq = tf.constant(visualize_freq, tf.int64)
        self._scale_regressor = generator.define_regressor()
        if initial_weights is not None:
            self._scale_regressor.load_weights(os.path.join(initial_weights, 'scale_gen.h5'))
        self._patcher = Patcher(self._patch, self._scale_regressor, min_patch_area=min_patch_area,
                                name='Patcher')
        self._images = None
        self._labels = None
        self.cur_step = None
        self.tb = None
        self._trainable_variables = self._scale_regressor.trainable_variables + self._patch.trainable_variables

        iou = self.config.nms_configs.iou_thresh
        self._metric = metrics.AttackSuccessRate(iou_thresh=iou)

    def filter_valid_boxes(self, boxes):
        boxes_h = boxes[:, :, 2] - boxes[:, :, 0]
        boxes_w = boxes[:, :, 3] - boxes[:, :, 1]
        boxes_area = boxes_h * boxes_w
        return tf.greater(boxes_area, tf.constant(400.))

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
        return boxes

    def second_pass(self, images):
        with tf.name_scope('attack_pass'):
            cls_outputs, box_outputs = self.model(images, pre_mode=None, post_mode=None)
            cls_outputs = postprocess.to_list(cls_outputs)
            box_outputs = postprocess.to_list(box_outputs)
            boxes, scores, classes = postprocess.pre_nms(self.config.as_dict(), cls_outputs, box_outputs)
            person_indices = tf.equal(classes, tf.constant(0))  # taking postprocess.CLASS_OFFSET into account
            scores = tf.ragged.boolean_mask(scores, person_indices)

            boxes = tf.ragged.boolean_mask(boxes, person_indices)
            classes = tf.ragged.boolean_mask(classes, person_indices)

            # valid_boxes = self.filter_valid_boxes(boxes)
            # scores = tf.ragged.boolean_mask(scores, valid_boxes)
            # boxes = tf.ragged.boolean_mask(boxes, valid_boxes)
            # classes = tf.ragged.boolean_mask(classes, valid_boxes)
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
        boxes = self._labels = self.first_pass(images)

        with tf.GradientTape() as tape:
            self._images, scale_losses = self._patcher([boxes, images], training=training)
            boxes_pred, scores, classes = self.second_pass(self._images)
            sc_losses = tf.reduce_max(scores, axis=1)
            loss = tf.reduce_sum(sc_losses + (sc_losses - scale_losses) ** 2.)

        self.add_loss(loss)
        self.add_loss(tf.reduce_sum(scale_losses))
        self.add_loss(tf.reduce_sum(sc_losses))

        boxes_pred, scores_pred = self._postprocessing(boxes_pred, scores, classes)
        self.add_metric(self._metric(boxes_pred, boxes), name='mean_asr')

        func = functools.partial(self.vis_images, boxes_pred, training)
        tf.cond(tf.equal(tf.math.floormod(self.cur_step, self.visualize_freq), tf.constant(0, tf.int64)),
                func, lambda: None)

        if training:
            return tape.gradient(loss, self._trainable_variables)

        return boxes_pred, scores_pred

    def vis_images(self, boxes_pred, training):
        images, labels = self._images, self._labels
        _, h, w, _ = tf.unstack(tf.cast(tf.shape(images), tf.float32))

        def convert_format(box):
            ymin, xmin, ymax, xmax = tf.unstack(box.to_tensor(), axis=2)
            return tf.stack([ymin / h, xmin / w, ymax / h, xmax / w], axis=2)

        labels = convert_format(labels)
        boxes_pred = convert_format(boxes_pred)
        images = tf.image.draw_bounding_boxes(images, labels, tf.constant([[0., 1., 0.]]))
        images = tf.image.draw_bounding_boxes(images, boxes_pred, tf.constant([[0., 0., 1.]]))
        images = tf.cast(images * self.config.stddev_rgb + self.config.mean_rgb, tf.uint8)
        tr = 'train' if training else 'val'
        with self.tb._writers[tr].as_default():
            if training:
                idx = tf.reshape(tf.stack(tf.meshgrid(tf.range(0., 1.01, .1), tf.range(0., 1.01, .1), indexing='ij'),
                                          axis=-1), (121, 2))
                scales = self._scale_regressor(idx)
                tf.summary.image('Patch Scales', tf.reshape(scales, (1, 11, 11, 1)), step=self.cur_step)

            tf.summary.image('Sample', images, step=self.cur_step, max_outputs=tf.shape(self._images)[0])

    def train_step(self, inputs):
        self.cur_step = self.tb._train_step
        self.reset_state()
        grads = self(inputs)
        self.optimizer.apply_gradients([*zip(grads, self._trainable_variables)])
        return self.update_metrics()

    def test_step(self, inputs):
        self.cur_step = self.tb._val_step
        self.reset_state()
        self(inputs, training=False)
        return self.update_metrics()

    def update_metrics(self):
        ret = {'loss': self.losses[0], 'scale_loss': self.losses[1], 'score_loss': self.losses[2]}
        ret.update({m.name: m.result() for m in self.metrics})
        return ret

    def reset_state(self):
        self._images = self._labels = None

    def save_weights(self, dirpath, **kwargs):
        os.makedirs(dirpath)
        self._scale_regressor.save(os.path.join(dirpath, 'scale_gen'), save_format='tf')
        self._patch.save(os.path.join(dirpath, 'gen'), save_format='tf')


class Patcher(tf.keras.layers.Layer):
    def __init__(self, patch: tf.keras.Model, scale_regressor: tf.keras.Model, *args, aspect=1.,
                 origin=(.5, .5), min_patch_area=60, **kwargs):
        super().__init__(*args, trainable=False, **kwargs)
        self._patch_gen = patch
        self._batch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._patch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._boxes = None
        self._patches = None
        self.aspect = aspect
        self.origin = origin
        self.min_patch_area = min_patch_area
        self._matcher = histogram_matcher.HistogramMatcher(name='Hist_Matcher')
        self._scale_regressor = scale_regressor

    def add_patches_to_image(self, image, training=False):
        h, w, _ = tf.unstack(tf.cast(tf.shape(image), tf.float32))
        boxes = self._boxes[self._batch_counter]
        patch = self._patches[self._batch_counter]
        patch = self._matcher((patch, image))

        def process():
            nonlocal image
            boxes_h, boxes_w = (boxes[:, 2] - boxes[:, 0]) / h, (boxes[:, 3] - boxes[:, 1]) / w
            scales = self._scale_regressor(tf.stack([boxes_h, boxes_w], axis=-1), training=training)

            patch_boxes = tf.vectorized_map(functools.partial(self.create, image), tf.concat([boxes, scales], axis=1))
            valid_indices = tf.where(tf.greater(patch_boxes[:, 2] * patch_boxes[:, 3],
                                                tf.constant(self.min_patch_area, tf.float32)))
            patch_boxes = tf.gather_nd(patch_boxes, valid_indices)
            scales = tf.gather_nd(scales, valid_indices)

            self._patch_counter.assign(tf.constant(0))
            loop_fn = functools.partial(self.add_patch_to_image, patch_boxes, patch)
            image, _ = tf.while_loop(lambda image, j: tf.less(self._patch_counter, tf.shape(patch_boxes)[0]),
                                     loop_fn, [image, self._patch_counter])

            scale_loss = tf.reduce_max(scales)
            return image, scale_loss

        self._batch_counter.assign_add(tf.constant(1))
        return tf.cond(tf.equal(tf.shape(boxes)[0], 0), lambda: (image, 0.), process)

    def add_patch_to_image(self, patch_boxes, patch, image, j):
        ymin_patch, xmin_patch, patch_h, patch_w = tf.unstack(tf.cast(patch_boxes[self._patch_counter], tf.int32))
        ymax = ymin_patch + patch_h
        xmax = xmin_patch + patch_w
        idx = tf.stack(tf.meshgrid(tf.range(ymin_patch, ymax), tf.range(xmin_patch, xmax), indexing='ij'), axis=-1)

        im = tf.image.resize(patch, tf.stack([patch_h, patch_w]))
        im = tf.image.grayscale_to_rgb(im)

        image = tf.tensor_scatter_nd_update(image, idx, im)
        self._patch_counter.assign_add(tf.constant(1))
        return [image, self._patch_counter]

    @staticmethod
    def create(image, item):
        ymin, xmin, ymax, xmax, scale = tf.unstack(item, 5)

        h = ymax - ymin
        w = xmax - xmin

        area = h * w
        patch_size = tf.floor(tf.sqrt(area * scale))

        patch_w = patch_size
        patch_h = patch_size

        orig_y = ymin + h / 2.
        orig_x = xmin + w / 2.

        ymin_patch = tf.maximum(orig_y - patch_h / 2., 0.)
        xmin_patch = tf.maximum(orig_x - patch_w / 2., 0.)

        shape = tf.cast(tf.shape(image), tf.float32)
        ymin_patch = tf.cond(tf.greater(ymin_patch + patch_h, shape[0]),
                             lambda: shape[0] - patch_h, lambda: ymin_patch)
        xmin_patch = tf.cond(tf.greater(xmin_patch + patch_w, shape[1]),
                             lambda: shape[1] - patch_w, lambda: xmin_patch)

        return tf.stack([ymin_patch, xmin_patch, patch_h, patch_w])

    def call(self, inputs, training=False):
        self._boxes, images = inputs
        self._patches = self._patch_gen(images)
        self._batch_counter.assign(tf.constant(0))
        images, losses = tf.map_fn(functools.partial(self.add_patches_to_image, training=training), images,
                                   fn_output_signature=(tf.TensorSpec(shape=images.shape[0], dtype=tf.float32),
                                                        tf.float32))
        return images, losses


def get_victim_model(download_model=False):
    if download_model:
        # Download checkpoint.
        util.download(MODEL)

    logger.info(f'Using model in {MODEL}')
    driver = infer_lib.KerasDriver(MODEL, debug=False, model_name=MODEL)
    return driver.model


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
    save_file = 'patch_{epoch:02d}_{val_mean_asr:.4f}.h5'
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=100,
                        steps_per_epoch=20,  # train_len,
                        validation_steps=20,  # val_len,
                        callbacks=[tb_callback,
                                   tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, save_file),
                                                                      monitor='val_mean_asr',
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
