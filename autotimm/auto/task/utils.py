"""Utils for auto tasks."""
import copy
import warnings

import autogluon.core as ag

from ..estimators import ImageClassificationEstimator
from ..estimators.base_estimator import BaseEstimator


class ConfigDict(dict):
    """The view of a config dict where keys can be accessed like attribute, it
    also prevents naive modifications to the key-values.

    Parameters
    ----------
    config : dict
        The configuration dict.

    Attributes
    ----------
    __dict__ : type
        The internal config as a `__dict__`.
    """
    MARKER = object()

    def __init__(self, value=None):
        super(ConfigDict, self).__init__()
        self.__dict__['_freeze'] = False
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict, given {}'.format(type(value)))
        self.freeze()

    def freeze(self):
        self.__dict__['_freeze'] = True

    def is_frozen(self):
        return self.__dict__['_freeze']

    def unfreeze(self):
        self.__dict__['_freeze'] = False

    def __setitem__(self, key, value):
        if self.__dict__.get('_freeze', False):
            msg = (
                'You are trying to modify the config to "{}={}" after initialization, '
                ' this may result in unpredictable behaviour'.format(
                    key, value))
            warnings.warn(msg)
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        super(ConfigDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, ConfigDict.MARKER)
        if found is ConfigDict.MARKER:
            if self.__dict__['_freeze']:
                raise KeyError(key)
            found = ConfigDict()
            super(ConfigDict, self).__setitem__(key, found)
        if isinstance(found, ConfigDict):
            found.__dict__['_freeze'] = self.__dict__['_freeze']
        return found

    def __setstate__(self, state):
        vars(self).update(state)

    def __getstate__(self):
        return vars(self)

    __setattr__, __getattr__ = __setitem__, __getitem__


def auto_suggest(config, estimator, logger):
    """Automatically suggest some hyperparameters based on the dataset
    statistics."""
    # specify estimator search space
    if estimator is None:
        estimator_init = []
        config['estimator'] = ag.Categorical(*estimator_init)
    elif isinstance(estimator, str):
        named_estimators = {'img_cls': ImageClassificationEstimator}
        if estimator.lower() in named_estimators:
            estimator = [named_estimators[estimator.lower()]]
        else:
            available_ests = named_estimators.keys()
            raise ValueError(
                f'Unknown estimator name: {estimator}, options: {available_ests}'
            )
    elif isinstance(estimator, (tuple, list)):
        pass
    else:
        if isinstance(estimator, ag.Space):
            estimator = estimator.data
        elif isinstance(estimator, str):
            estimator = [estimator]
        if not estimator:
            raise ValueError(
                'Unable to determine the estimator for fit function.')
        if len(estimator) == 1:
            config['estimator'] = estimator[0]
        else:
            config['estimator'] = ag.Categorical(*estimator)


def get_recursively(search_dict, field):
    """Takes a dict with nested dicts, and searches all dicts for a key of the
    field provided."""
    fields_found = []

    for key, value in search_dict.items():

        if key == field:
            fields_found.append(value)

        elif isinstance(value, dict):
            results = get_recursively(value, field)
            for result in results:
                fields_found.append(result)

    return fields_found


def config_to_nested(config):
    """Convert config to nested version."""
    estimator = config.get('estimator', None)
    transfer = config.get('transfer', None)
    # choose hyperparameters based on pretrained model in transfer learning
    if transfer:
        # choose estimator
        estimator = ImageClassificationEstimator
    elif isinstance(estimator, str):
        if estimator == 'img_cls':
            estimator = ImageClassificationEstimator
        else:
            raise ValueError(f'Unknown estimator: {estimator}')
    else:
        assert issubclass(estimator, BaseEstimator)

    cfg_map = estimator._default_cfg.asdict()

    def _recursive_update(config, key, value, auto_strs, auto_ints):
        for k, v in config.items():
            if k in auto_strs:
                config[k] = 'auto'
            if k in auto_ints:
                config[k] = -1
            if key == k:
                config[key] = value
            elif isinstance(v, dict):
                _recursive_update(v, key, value, auto_strs, auto_ints)

    auto_strs = ['data_dir', 'dataset', 'dataset_root']
    auto_ints = ['num_training_samples']
    for k, v in config.items():
        _recursive_update(cfg_map, k, v, auto_strs, auto_ints)
    cfg_map['estimator'] = estimator
    return cfg_map
