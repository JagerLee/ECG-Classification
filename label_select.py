import numpy as np
import pandas as pd
import os
import sys
import argparse
from data_preprocess import get_files_list, source_path


def get_args():
    parser = argparse.ArgumentParser(description='select label above threshold.')

    parser.add_argument('-th', '--threshold', dest='threshold', type=float, default=0.99,
                        help='Threshold of label select. Default: 0.99')

    return parser.parse_args()
    

def label_select(threshold):
    label_df = pd.read_csv(os.path.join(source_path, 'hf_round2_arrythmia.txt'), header=None, sep='\t\t\t',
                           engine='python', encoding='utf-8')
    label_dict = {}
    for i, label in enumerate(label_df[0]):
        label_dict[label] = i
    label_df = pd.read_excel(os.path.join(source_path, 'hf_round2_train.xlsx'), header=None)
    files_list = get_files_list(os.path.join(source_path, 'hf_round2_train'))
    y_t = np.zeros((len(label_dict),))
    for i, file in enumerate(files_list):
        df1 = label_df.loc[label_df[0] == file, 3:]
        if not df1.empty:
            for label in df1.values[0]:
                if type(label) == str:
                    y_t[int(label_dict[label])] += 1.0
    label_idx = np.argsort(-y_t)
    label_sort = y_t[label_idx]
    total_num = np.sum(y_t)
    tmp_sum = 0
    for i, num in enumerate(label_sort):
        tmp_sum += num
        if tmp_sum / total_num > threshold:
            break
    label_select = label_idx[:i + 1]
    with open(os.path.join(source_path, 'label_select.txt'), 'w', encoding='utf-8') as f:
        for i in label_select:
            for key in label_dict:
                if label_dict[key] == i:
                    f.write(key + '\t\t\t\n')


if __name__ == '__main__':
    threshold = get_args().threshold # 筛选标签比例
    label_select(threshold)
