'''
Author: jianzhnie
Date: 2021-11-29 14:07:24
LastEditTime: 2021-11-29 14:24:09
LastEditors: jianzhnie
Description:
'''
import argparse
import os
import shutil

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model for different kaggle competitions.')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='',
        help='training and validation pictures to use.')
    parser.add_argument(
        '--dataset', type=str, default='dog', help='the kaggle competition')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()

    if opt.dataset == 'dog':
        csvfile = 'labels.csv'
        pic_path = 'images_all/'
        train_path = 'split/train/'
        test_path = 'split/test/'
        split_ratio = 0.8

        csvfile = os.path.join(opt.data_dir, 'dog-breed-identification',
                               csvfile)
        pic_path = os.path.join(opt.data_dir, 'dog-breed-identification',
                                pic_path)
        train_path = os.path.join(opt.data_dir, 'dog-breed-identification',
                                  train_path)
        test_path = os.path.join(opt.data_dir, 'dog-breed-identification',
                                 test_path)

        csvfile = open(csvfile, 'r')
        data = []
        for line in csvfile:
            data.append(list(line.strip().split(',')))
        for i in range(len(data)):
            if i == 0:
                continue
            if i >= 1:
                cl = data[i][1]
                name = data[i][0]
                path = pic_path + str(name) + '.jpg'
                isExists = os.path.exists(path)
                if (isExists):
                    if not os.path.exists(train_path + cl):
                        os.makedirs(train_path + cl)
                    if not os.path.exists(test_path + cl):
                        os.makedirs(test_path + cl)
                    new_train_path = train_path + cl + '/' + str(name) + '.jpg'
                    new_eval_path = test_path + cl + '/' + str(name) + '.jpg'
                    if np.random.rand() < split_ratio:
                        shutil.copyfile(path, new_train_path)
                        print(str(name), 'train', 'success')
                    else:
                        shutil.copyfile(path, new_eval_path)
                        print(str(name), 'eval', 'success')
                else:
                    print(str(name) + ',not here')


if __name__ == '__main__':
    main()
