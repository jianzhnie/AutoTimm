'''
Author: jianzhnie
Date: 2021-08-09 15:54:25
LastEditTime: 2021-11-29 15:32:28
LastEditors: jianzhnie
Description:

'''
import autogluon.core as ag

from autotimm.auto import ImagePredictor

if __name__ == '__main__':
    train_dataset, valid_dataset, test_dataset = ImagePredictor.Dataset.from_folders(
        '/media/robin/DATA/datatsets/image_data/shopee-iet/images',
        test='None')
    predictor = ImagePredictor(log_dir='work_dir/checkpoint')

    predictor.fit(
        train_data=train_dataset,
        tuning_data=valid_dataset,
        hyperparameters={
            'model': ag.Categorical('resnet18'),
            'batch_size': ag.Categorical(8),
            'lr': ag.Categorical(0.001, 0.005, 0.0005, 0.0001),
            'epochs': 1,
            'cleanup_disk': False
        },
        hyperparameter_tune_kwargs={
            'num_trials': 4,
            'max_reward': 1.0,
            'searcher': 'random'
        },
        nthreads_per_trial=8,
        ngpus_per_trial=1,
        holdout_frac=0.1
    )  # you can trust the default config, we reduce the # epoch to save some build time

    test_acc, _ = predictor.evaluate(test_dataset)
    print(f'Test Accuracy: {test_acc}')
    res = predictor.predict(data=test_dataset, batch_size=32)

    res_ = predictor.predict(data=test_dataset, batch_size=32)
    res_.to_csv('./result.csv')
    print('*' * 10)
    print('result saved!')
