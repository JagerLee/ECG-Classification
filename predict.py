import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
import pandas as pd
import json
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='predict ecg data to label.')

    parser.add_argument('-m', '--model', dest='model_path', type=str,
                        help='Model path of predict.')
    parser.add_argument('-i', '--input', dest='input_path', type=str,
                        help='Input file or dictionary path of predict.')

    return parser.parse_args()


def get_files_list(root):
    files_list = []
    for root, dirs, files in os.walk(root):
        files_list += files

    return files_list


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


class ECG_Classifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self._loadModel()
        self.label_dict = self._load_label_dict()

    def predict(self, input_path):
        if os.path.isdir(input_path):
            files_list = get_files_list(input_path)
            x = np.ones((len(files_list), 5000, 8))
            for i, file in enumerate(files_list):
                df = pd.read_csv(os.path.join(input_path, file), header=0, sep=' ')
                x[i] = df.values.astype(float)
        else:
            files_list = [input_path]
            x = np.ones((1, 5000, 8))
            df = pd.read_csv(input_path, header=0, sep=' ')
            x[0] = df.values.astype(float)
        y = self._predict(x)
        with open('data/predict/output.txt', 'w', encoding='utf-8') as f:
            for i, file in enumerate(files_list):
                f.write(file)
                for j, p in enumerate(y[i]):
                    if p > 0.5:
                        label = list(filter(lambda z: j == z[1], self.label_dict.items()))
                        f.write('\t' + label[0][0])
                f.write('\n')

    def _load_label_dict(self):
        with open(os.path.join(self.model_path, 'label_dict.json'), encoding='utf-8') as f:
            label_dict = json.load(f)

        return label_dict

    def _loadModel(self):
        model = keras.models.load_model(os.path.join(self.model_path, 'model.h5'),
                                        custom_objects={'precision': precision, 'recall': recall, 'f1_score': f1_score})

        return model

    def _predict(self, x):
        x_train_mean = np.load(os.path.join(self.model_path, 'x_train_mean.npy'))
        x_train_std = np.load(os.path.join(self.model_path, 'x_train_std.npy'))
        x = (x - x_train_mean) / x_train_std
        x = x.reshape(x.shape + (1,))

        return self.model.predict(x, batch_size=8)


if __name__ == '__main__':
    args = get_args()
    model_path = args.model_path
    input_path = args.input_path
    ecg = ECG_Classifier(model_path)
    ecg.predict(input_path)
