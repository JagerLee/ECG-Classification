from tensorflow import keras
import numpy as np
from train import precision, recall, f1_score


def precision1(y_true, y_pred):
    threshold = .5
    y_ = np.round(y_pred - threshold + .5)
    pred_pos = np.sum(y_)
    true_pos = np.sum((y_ + y_true) == 2)

    return true_pos / (pred_pos + 1e-7)


def recall1(y_true, y_pred):
    threshold = .5
    y_ = np.round(y_pred - threshold + .5)
    real_pos = np.sum(y_true)
    true_pos = np.sum((y_ + y_true) == 2)

    return true_pos / (real_pos + 1e-7)


def f1_score1(y_true, y_pred):
    threshold = .5
    y_ = np.round(y_pred - threshold + .5)
    pred_pos = np.sum(y_)
    real_pos = np.sum(y_true)
    true_pos = np.sum((y_ + y_true) == 2)
    prec = true_pos / (pred_pos + 1e-7)
    rec = true_pos / (real_pos + 1e-7)

    return 2 * prec * rec / (prec + rec + 1e-7)


def loadModel(x):
    x_train_mean = np.load('save_model/x_train_mean.npy')
    x_train_std = np.load('save_model/x_train_std.npy')
    x = (x - x_train_mean) / x_train_std
    x = x.reshape(x.shape + (1,))
    model = keras.models.load_model('save_model/model.h5',
                                    custom_objects={'precision': precision, 'recall': recall, 'f1_score': f1_score})
    y = model.predict(x, batch_size=8)

    return y


def evaluate():
    x_test = np.load('data/test/x_test.npy')
    y_true = np.load('data/test/y_test.npy')
    y_pred = loadModel(x_test)
    prec = precision1(y_true, y_pred)
    rec = recall1(y_true, y_pred)
    print('precision: ', prec)
    print('recall: ', rec)
    print('f1_score: ', 2 * prec * rec / (prec + rec + 1e-7))


if __name__ == '__main__':
    evaluate()