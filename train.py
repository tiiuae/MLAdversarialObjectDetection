"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: June 15, 2022

Purpose: replaces jupyter based training
"""


def allow_direct_imports_from(dirname):
    import sys
    if dirname not in sys.path:
        sys.path.append(dirname)

allow_direct_imports_from('automl/efficientdet')

import os
import tensorflow as tf

import dynamicattacker as attacker
import train_data_generator
import util

MODEL = 'efficientdet-lite4'


def main(download_model=False):
    log_dir = util.ensure_empty_dir('log_dir/atk_new_data')
    gpu = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)

    victim_model = util.get_victim_model(MODEL, download_model)
    config_override = {'nms_configs': {'iou_thresh': .5, 'score_thresh': .5}}
    model = attacker.DynamicPatchAttacker(victim_model,
                                          # initial_weights='save_dir/patch_04_0.5024',
                                          config_override=config_override,
                                          visualize_freq=200)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), run_eagerly=False)

    datasets: dict = train_data_generator.partition(model.config, 'train_eval', '', batch_size=12, shuffle=True)

    train_ds = datasets['train']['dataset']
    val_ds = datasets['val']['dataset']
    train_len = datasets['train']['length']
    val_len = datasets['val']['length']
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir, write_graph=False,
                                                 write_steps_per_second=True,
                                                 update_freq='epoch')
    model.tb = tb_callback

    save_dir = util.ensure_empty_dir('save_dir_new_data')
    save_file = 'patch_{epoch:02d}_{val_asr_to_scale:.4f}'
    model.fit(train_ds, validation_data=val_ds, epochs=500, steps_per_epoch=train_len,
              # initial_epoch=12,
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
                         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=.5, min_lr=1e-5)
                         ])


if __name__ == '__main__':
    main()
