import pandas as pd
import numpy as np
import os
import json
import argparse


source_path = 'data/source_data/'


def get_args():
    parser = argparse.ArgumentParser(description='data preprocess of ecg.')

    parser.add_argument('-t', '--test', dest='test_size', type=float, default=0.2,
                        help='Test size of data set split. Default: 0.2')

    return parser.parse_args()


def get_files_list(root):
    files_list = []
    for root, dirs, files in os.walk(root):
        files_list += files

    return files_list


def get_label_dict(path):
    label_df = pd.read_csv(path, encoding='utf-8', header=None, sep='\t\t\t', engine='python')
    label_dict = {}
    for i, label in enumerate(label_df[0]):
        label_dict[label] = i
    if not os.path.exists('save_model'):
        os.mkdir('save_model')
    with open('save_model/label_dict.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(label_dict, ensure_ascii=False))


def train_test_split(test_size=.2):
    x_all = np.load(os.path.join(source_path, 'x_all.npy'))
    y_all = np.load(os.path.join(source_path, 'y_all.npy'))
    total_num = y_all.shape[0]
    test_num = int(test_size * total_num)
    shuffle_idx = np.arange(total_num)
    np.random.shuffle(shuffle_idx)
    x_all = x_all[shuffle_idx]
    y_all = y_all[shuffle_idx]

    x_train = x_all[:total_num - test_num]
    y_train = y_all[:total_num - test_num]
    x_test = x_all[total_num - test_num:]
    y_test = y_all[total_num - test_num:]

    if not os.path.exists('data/train'):
        os.mkdir('data/train')
    if not os.path.exists('data/test'):
        os.mkdir('data/test')
    print('train set shape: ', x_train.shape, y_train.shape)
    np.save('data/train/x_train.npy', x_train)
    np.save('data/train/y_train.npy', y_train)
    print('test set shape: ', x_test.shape, y_test.shape)
    np.save('data/test/x_test.npy', x_test)
    np.save('data/test/y_test.npy', y_test)


def get_unique_idx(x_t):
    x_t = x_t[:, :, 0:3]
    unique_idx = [0]
    print(x_t.shape)
    x_sum = x_t.sum(1)
    for i in range(x_t.shape[0]):
        is_unique = 1
        for idx in unique_idx:
            if (x_sum[i] == x_sum[idx]).all():
                is_unique = 0
                break
        if is_unique:
            unique_idx.append(i)

    return unique_idx


def txt2npy():
    data_path = os.path.join(source_path, 'hf_round2_train')
    t_label_path = os.path.join(source_path, 'hf_round2_train.xlsx')

    files_list = get_files_list(data_path)
    files_len = len(files_list)

    x_t = np.ones((files_len, 5000, 8))
    for i, file in enumerate(files_list):
        df = pd.read_csv(os.path.join(data_path, file), header=0, sep=' ')
        x_t[i] = df.values.astype(float)

    t_label_df = pd.read_excel(t_label_path, header=None)
    with open('save_model/label_dict.json', encoding='utf-8') as f:
        label_dict = json.load(f)
    y_t = np.zeros((files_len, len(label_dict)))
    for i, file in enumerate(files_list):
        df1 = t_label_df.loc[t_label_df[0] == file, 3:]
        if not df1.empty:
            for label in df1.values[0]:
                for item in label_dict.items():
                    if label == item[0]:
                        y_t[i, int(item[1])] = 1.0

    unique_idx = get_unique_idx(x_t)
    x_t = x_t[unique_idx]
    y_t = y_t[unique_idx]
    print('x shape: ', x_t.shape)
    np.save(os.path.join(source_path, 'x_all.npy'), x_t)
    print('y shape: ', y_t.shape)
    np.save(os.path.join(source_path, 'y_all.npy'), y_t)


def data_preprocess():
    test_size = get_args().test_size
    get_label_dict(os.path.join(source_path, 'label_select.txt'))
    txt2npy()
    train_test_split(test_size)


if __name__ == "__main__":
    data_preprocess()