import os
import autogluon.core as ag

def gluon_config_choice(dataset, model_choice="default"):

    custom_config = {
        'big_models': {
            'hyperparameters': {
                'model': ag.Categorical('resnet152_v1d', 'efficientnet_b4'),
                'lr':  ag.Categorical(0.001, 0.01, 0.1),
                'batch_size': ag.Categorical(16, 32),
                'epochs': 60,
                'early_stop_patience': -1,
                'ngpus_per_trial': 4,
                'cleanup_disk': False
            },
            'hyperparameter_tune_kwargs': {
                'num_trials': 48,
                'max_reward': 1.0,
                'searcher': 'random'
            },
            'time_limit': 3*24*3600
        },

        'search_models': {
            'hyperparameters': {
                'model': ag.Categorical('resnet152_v1d', 'efficientnet_b4'), # 'resnet152_v1d', 'efficientnet_b4', 'resnet152_v1d', 'efficientnet_b2', 
                'lr':  ag.Categorical(6e-2, 1e-1, 3e-1, 6e-1),
                'batch_size': ag.Categorical(32),
                'epochs': 50,
                'early_stop_patience': 10,
                'ngpus_per_trial': 4,
                'cleanup_disk': False
            },
            'hyperparameter_tune_kwargs': {
                'num_trials': 8,
                'max_reward': 1.0,
                'searcher': 'random'
            },
            'time_limit': 8*24*3600
        },

        'best_quality': {
            'hyperparameters': {
                'model': ag.Categorical('resnet50_v1b', 'resnet101_v1d', 'resnest200'),
                'lr': ag.Real(1e-5, 1e-2, log=True),
                'batch_size': ag.Categorical(8, 16, 32, 64, 128),
                'epochs': 120,
                'early_stop_patience': 10,
                'ngpus_per_trial': 4,
                'cleanup_disk': False
            },
            'hyperparameter_tune_kwargs': {
                'num_trials': 256,
                'searcher': 'bayesopt',
                'max_reward': 1.0,
            },
            'time_limit': 12*3600
        },

        'good_quality_fast_inference': {
            'hyperparameters': {
                'model': ag.Categorical('resnet50_v1b', 'resnet34_v1b'),
                'lr': ag.Real(1e-4, 1e-2, log=True),
                'batch_size': ag.Categorical(32, 64, 128),
                'epochs': 100,
                'early_stop_patience': 10,
                'ngpus_per_trial': 4,
                'cleanup_disk': False,
                },
            'hyperparameter_tune_kwargs': {
                'num_trials': 128,
                'max_reward': 1.0,
                'searcher': 'bayesopt',
                },
            'time_limit': 8*3600
        },

        'default_hpo': {
            'hyperparameters': {
                'model':  ag.Categorical('resnet50_v1b'),
                'lr': ag.Categorical(0.01, 0.005, 0.001, 0.02),
                'batch_size': ag.Categorical(32, 64, 128),
                'epochs': 50,
                'early_stop_patience': 10,
                'ngpus_per_trial': 4,
                'cleanup_disk': False,
                'final_fit': False,
            },
            'hyperparameter_tune_kwargs': {
                'num_trials':  12,
                'searcher': 'random',
                'max_reward': 1.0,
            },
            'time_limit': 16*3600
        },

        'default': {
            'hyperparameters': {
                'model': 'resnet18',
                'lr': 0.01,
                'batch_size': 64,
                'epochs': 10,
                'early_stop_patience': 10,
                'ngpus_per_trial': 4,
                'cleanup_disk': False,
                'final_fit': False,
                },
            'hyperparameter_tune_kwargs': {
                'num_trials': 1,
                'max_reward': 1.0,
                },
            'time_limit': 6*3600
        },

        'medium_quality_faster_inference': {
            'hyperparameters': {
                'model': ag.Categorical('resnet18_v1b', 'mobilenetv3_small'),
                'lr': ag.Categorical(0.01, 0.005, 0.001),
                'batch_size': ag.Categorical(64, 128),
                'epochs': ag.Categorical(50, 100),
                'early_stop_patience': 10,
                'ngpus_per_trial': 4,
                'cleanup_disk': False
            },
            'hyperparameter_tune_kwargs': {
                'num_trials': 32,
                'max_reward': 1.0,
                'searcher': 'bayesopt',
                },
            'time_limit': 6*3600
        }
    }
    config = custom_config[model_choice]
    return config
