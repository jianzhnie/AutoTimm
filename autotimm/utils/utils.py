import csv
import json
import os
import time

import yaml


def mkdir(root_dir):
    """mkdir."""
    tm = time.strftime('%Y%m%d-%H%M', time.localtime())
    file_path = os.path.join(root_dir, tm)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    else:
        print('%s has exists', file_path)
    return file_path


def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as yaml_file:
        result = yaml.load(yaml_file.read(), Loader=yaml.SafeLoader)
        return result


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result


def update_kwargs(hyperparameters, hyperparameter_tune_kwargs, summary):
    for key in hyperparameters:
        hyperparameters[key] = summary[key]
    hyperparameter_tune_kwargs['num_trials'] = 1
    return hyperparameters, hyperparameter_tune_kwargs


def is_valid_dir(logdir, trial_name):
    flag = False
    trial_dir = os.path.join(logdir, trial_name)
    if trial_name.startswith('.trial_') and os.path.isdir(trial_dir):
        flag = True
    return flag


def find_best_model(checkpoint_dir,
                    valid_summary_file='fit_summary_img_cls.ag'):
    _BEST_CHECKPOINT_FILE = 'best_checkpoint.pkl'
    _BEST_CONFIG_FILE = 'config.yaml'
    best_checkpoint = ''
    best_config = ''
    best_acc = -1
    result = {}
    if os.path.isdir(checkpoint_dir):
        for cb in os.listdir(checkpoint_dir):
            log_dir = os.path.join(checkpoint_dir, cb)
            if os.path.isdir(log_dir):
                trial_dirs = [
                    d for d in os.listdir(log_dir) if is_valid_dir(log_dir, d)
                ]
                for dd in trial_dirs:
                    try:
                        with open(
                                os.path.join(log_dir, dd, valid_summary_file),
                                'r') as f:
                            result = json.load(f)
                            acc = result.get('valid_acc', -1)
                            print('=' * 30, 'Check history trials results: ',
                                  log_dir, dd, acc)
                            if acc > best_acc and os.path.isfile(
                                    os.path.join(log_dir, dd,
                                                 _BEST_CHECKPOINT_FILE)):
                                best_checkpoint = os.path.join(
                                    log_dir, dd, _BEST_CHECKPOINT_FILE)
                                best_config = os.path.join(
                                    log_dir, dd, _BEST_CONFIG_FILE)
                                print('=' * 30, 'Find a Better checkpoint : ',
                                      best_checkpoint)
                                print('=' * 30, 'Find a Better config : ',
                                      best_config)
                                best_acc = acc
                    except IOError:
                        print('Error, can not find the vaild pth')
    return best_checkpoint, best_config, result


def find_best_model_loop(checkpoint_dir,
                         valid_summary_file='fit_summary_img_cls.ag'):
    _BEST_CHECKPOINT_FILE = 'best_checkpoint.pkl'
    _BEST_CONFIG_FILE = 'config.yaml'
    best_checkpoint = ''
    best_config = ''
    best_acc = -1
    result = {}
    for root, dirs, files in os.walk(checkpoint_dir):
        for trial_name in dirs:
            if trial_name.startswith('.trial_'):
                trial_dir = os.path.join(root, trial_name)
                try:
                    with open(
                            os.path.join(trial_dir, valid_summary_file),
                            'r') as f:
                        result = json.load(f)
                        acc = result.get('valid_acc', -1)
                        print('=' * 30, 'Check history trials results: ',
                              trial_dir, acc)
                        if acc > best_acc and os.path.isfile(
                                os.path.join(trial_dir,
                                             _BEST_CHECKPOINT_FILE)):
                            best_checkpoint = os.path.join(
                                trial_dir, _BEST_CHECKPOINT_FILE)
                            best_config = os.path.join(trial_dir,
                                                       _BEST_CONFIG_FILE)
                            print('=' * 30, 'Find a Better checkpoint : ',
                                  best_checkpoint)
                            print('=' * 30, 'Find a Better config : ',
                                  best_config)
                            best_acc = acc
                except IOError:
                    print('Error, can not find the vaild pth')
    return best_checkpoint, best_config, result


def write_csv_file(path, head, data):
    try:
        with open(path, 'a+', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if head is not None:
                writer.writerow(head)
            for row in data:
                writer.writerow(row)
            print('Write a CSV file to path %s Successful.' % path)
    except Exception as e:
        print('Write an CSV file to path: %s, Case: %s' % (path, e))


def parse_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = config.get('img_cls').get('model')
    batch_size = config.get('train').get('batch_size')
    epochs = config.get('train').get('epochs')
    learning_rate = config.get('train').get('lr')
    momentum = config.get('train').get('momentum')
    wd = config.get('train').get('wd')
    input_size = config.get('train').get('input_size')

    return model, batch_size, epochs, learning_rate, momentum, wd, input_size


if __name__ == '__main__':
    find_best_model_loop(
        '/home/robin/jianzh/automl/autodl/benchmark/hymenoptera')
