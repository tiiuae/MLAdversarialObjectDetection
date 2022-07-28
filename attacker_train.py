"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: June 15, 2022

Purpose: replaces jupyter based training for patch attacker
"""
import util
util.allow_direct_imports_from('automl/efficientdet')

import os
import tensorflow as tf

import attacker
import train_data_generator

MODEL = 'efficientdet-lite4'


def main():
    """start attack training"""
    #  setup resources
    log_dir = util.ensure_empty_dir('log_dir/1atk_new_tv')
    gpu = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)

    # get victim model and configure some hyperparams
    victim_model = util.get_victim_model(MODEL, download_model=False)
    config_override = {'nms_configs': {'iou_thresh': .5, 'score_thresh': .5}}

    # init attacker class.
    model = attacker.PatchAttacker(victim_model,
                                   # initial_weights='save_dir/patch_04_0.5024',
                                   config_override=config_override,
                                   visualize_freq=200)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), run_eagerly=False)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir, write_graph=False,
                                                 write_steps_per_second=True,
                                                 update_freq='epoch')
    model.tb = tb_callback

    # init datasets
    datasets: dict = train_data_generator.partition(model.config, 'train_eval', None, batch_size=12, shuffle=True,
                                                    filter_data=False)
    train_ds = datasets['train']['dataset']
    val_ds = datasets['val']['dataset']
    train_len = datasets['train']['length']
    val_len = datasets['val']['length']

    # init save dir
    save_dir = util.ensure_empty_dir('1save_dir_new_tv')
    save_file = 'patch_{epoch:02d}_{val_asr_to_scale:.4f}'

    # train
    model.fit(train_ds, validation_data=val_ds, epochs=500, steps_per_epoch=train_len,
              validation_steps=val_len,
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
                                                            ),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, factor=.5, min_lr=1e-4,
                                                              patience=50)
                         ])


if __name__ == '__main__':
    main()
