#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import argparse

np.random.seed(813306)


def get_args():
    parser = argparse.ArgumentParser(description='train ResNet from ecg data.')
    parser.add_argument('-b', '--batch',  dest='batch_size', type=int, default=8,
                        help='Batch size of training. Default: 8')
    parser.add_argument('-lr', '--lr', dest='lr', type=float, default=0.01,
                        help='Learning rate of training. Default: 0.01')
    parser.add_argument('-e', '--epoch', dest='epoch', type=int, default=10,
                        help='Max epoch of training. Default: 10')

    return parser.parse_args()


def precision(y_true, y_pred):
    threshold = .5
    y_ = K.round(y_pred - threshold + .5)
    pred_pos = K.sum(y_)
    true_pos = K.sum(K.cast_to_floatx((y_ + y_true) == 2))

    return true_pos / (pred_pos + 1e-7)


def recall(y_true, y_pred):
    threshold = .5
    y_ = K.round(y_pred - threshold + .5)
    real_pos = K.sum(y_true)
    true_pos = K.sum(K.cast_to_floatx((y_ + y_true) == 2))

    return true_pos / (real_pos + 1e-7)


def f1_score(y_true, y_pred):
    threshold = .5
    y_ = K.round(y_pred - threshold + .5)
    pred_pos = K.sum(y_)
    real_pos = K.sum(y_true)
    true_pos = K.sum(K.cast_to_floatx((y_ + y_true) == 2))
    prec = true_pos / (pred_pos + 1e-7)
    rec = true_pos / (real_pos + 1e-7)

    return 2 * prec * rec / (prec + rec + 1e-7)


def res_module(y, input_shape, n_feature_maps):
    # print('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps, (8, 3), 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    # print('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps, (5, 2), 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    # print('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps, (3, 2), 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps, 1, 1, padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    # print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    return y


def build_resnet(input_shape, n_feature_maps, nb_classes):
    x_input = keras.layers.Input(shape=(input_shape))
    conv_x = keras.layers.BatchNormalization()(x_input)

    seg_num = 10
    seg_len = int(input_shape[0] / seg_num)
    input_shape_segment = (seg_len, input_shape[1], input_shape[-1])
    y_segment = []
    for i in range(seg_num):
        y_list = []
        y = res_module(conv_x[:, i * seg_len: (i + 1) * seg_len], input_shape_segment, n_feature_maps / 2)
        # y = res_module(y, input_shape_segment, n_feature_maps)
        for i in range(input_shape[1]):
            y_ = keras.layers.Flatten()(y[:, :, i, :])
            y_ = keras.layers.Dense(128, activation='relu')(y_)
            y_ = keras.layers.Reshape(y_.shape[1:] + (1, 1,))(y_)
            y_list.append(y_)
        y = keras.layers.concatenate(y_list, 2)
        y_segment.append(y)

    y_segment = keras.layers.concatenate(y_segment, 1)
    shape = y_segment.shape
    y_segment = res_module(y_segment, shape, n_feature_maps)
    y_segment = res_module(y_segment, shape, n_feature_maps * 2)
    y_segment = keras.layers.GlobalAveragePooling2D()(y_segment)

    out = keras.layers.Dense(nb_classes, activation='sigmoid')(y_segment)
    print('        -- model was built.')
    return x_input, out


def loadData():
    x_train = np.load('data/train/x_train.npy')
    x_test = np.load('data/test/x_test.npy')
    y_train = np.load('data/train/y_train.npy')
    y_test = np.load('data/test/y_test.npy')

    return x_train, y_train, x_test, y_test


def normalization(x_train, x_test):
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / (x_train_std)
    x_test = (x_test - x_train_mean) / (x_train_std)

    return x_train, x_test, x_train_mean, x_train_std


def train(model, x_train, y_train, x_test, y_test):
    args = get_args()
    batch_size = args.batch_size
    nb_epochs = args.epoch
    lr = args.lr
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                  patience=50, min_lr=lr)
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs,
                     verbose=1, validation_data=(x_test, y_test), callbacks=[reduce_lr])
    log = pd.DataFrame(hist.history)
    print(log.loc[log['loss'].idxmin]['loss'],
          log.loc[log['loss'].idxmin]['val_precision'],
          log.loc[log['loss'].idxmin]['val_recall'],
          log.loc[log['loss'].idxmin]['val_f1_score'])

    return log


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = loadData()

    if not os.path.exists('data/predict'):
        os.mkdir('data/predict')
    if not os.path.exists('save_model'):
        os.mkdir('save_model')
    nb_classes = y_test.shape[1]
    x, y = build_resnet(x_train.shape[1:] + (1,), 64, nb_classes)
    model = keras.models.Model(inputs=x, outputs=y)
    optimizer = keras.optimizers.Adam()
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=[precision, recall, f1_score])

    x_train, x_test, x_train_mean, x_train_std = normalization(x_train, x_test)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    train(model, x_train, y_train, x_test, y_test)

    model.save('save_model/model' + '.h5')
    np.save('save_model/x_train_mean.npy', x_train_mean)
    np.save('save_model/x_train_std.npy', x_train_std)

