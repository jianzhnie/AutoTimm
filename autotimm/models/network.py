import copy

import torch.nn as nn
from timm.models.layers.classifier import ClassifierHead

from .model_inputsize import model_inputsize
from .model_zoo import get_model, get_model_list


def get_input_size(model_name):
    input_size = model_inputsize.get(model_name, 224)
    return input_size


def init_network(model_name, num_class, pretrained=False):
    if not num_class:
        raise ValueError(
            'Unable to create network when `num_class` is unknown. \
            It should be inferred from dataset or resumed from saved states.')
    if model_name:
        net = get_model(model_name, pretrained=pretrained)
    # reset last fully connected layer
    fc_layer_found = False
    for fc_name in ('fc', 'classifier', 'head', 'classif'):
        fc_layer = getattr(net, fc_name, None)
        if fc_layer is not None:
            fc_layer_found = True
            break

    if fc_layer_found:
        if isinstance(fc_layer, nn.Linear):
            in_features = fc_layer.in_features
            new_fc_layer = nn.Linear(in_features, num_class)
            setattr(net, fc_name, new_fc_layer)
        elif isinstance(fc_layer, ClassifierHead):
            head_fc = getattr(fc_layer, 'fc', None)
            assert head_fc is not None, 'Can not find the fc layer in ClassifierHead'
            if isinstance(head_fc, nn.Linear):
                in_features = head_fc.in_features
                new_fc = nn.Linear(in_features, num_class)
            elif isinstance(head_fc, nn.Conv2d):
                in_channels = head_fc.in_channels
                new_fc = nn.Conv2d(in_channels, num_class, kernel_size=1)
            setattr(fc_layer, 'fc', new_fc)
            setattr(net, fc_name, fc_layer)
        elif isinstance(fc_layer, nn.Conv2d):
            in_channels = fc_layer.in_channels
            new_fc_layer = nn.Conv2d(in_channels, num_class, kernel_size=1)
            setattr(net, fc_name, new_fc_layer)
        else:
            raise TypeError(
                f'Invalid FC layer type {type(fc_layer)} found, expected (Conv2d, Linear)...'
            )
    else:
        raise RuntimeError(
            'Unable to modify the last fc layer in network, (fc, classifier, ClassifierHead) expected...'
        )
    return net


def get_feature_net(net):
    """Get the network slice for feature extraction only."""
    feature_net = copy.copy(net)
    fc_layer_found = False
    for fc_name in ('fc', 'classifier', 'head', 'classif'):
        fc_layer = getattr(feature_net, fc_name, None)
        if fc_layer is not None:
            fc_layer_found = True
            break
    new_fc_layer = nn.Identity()
    if fc_layer_found:
        if isinstance(fc_layer, ClassifierHead):
            head_fc = getattr(fc_layer, 'fc', None)
            assert head_fc is not None, 'Can not find the fc layer in ClassifierHead'
            setattr(fc_layer, 'fc', new_fc_layer)
            setattr(feature_net, fc_name, fc_layer)

        elif isinstance(fc_layer, (nn.Linear, nn.Conv2d)):
            setattr(feature_net, fc_name, new_fc_layer)
        else:
            raise TypeError(
                f'Invalid FC layer type {type(fc_layer)} found, expected (Conv2d, Linear)...'
            )
    else:
        raise RuntimeError(
            'Unable to modify the last fc layer in network, (fc, classifier, ClassifierHead) expected...'
        )
    return feature_net


if __name__ == '__main__':
    model_names = get_model_list()
    # model_names = ['resnetv2_50x1_bitm']
    model_names = ['resnet18']
    for name in model_names:
        try:
            print('=' * 10, name)
            net = init_network(name, 2)
            feature_net = get_feature_net(net)
            fc_layer_found = False
            for fc_name in ('fc', 'classifier', 'head', 'classif'):
                fc_layer = getattr(feature_net, fc_name, None)
                if fc_layer is not None:
                    fc_layer_found = True
                    break
            if fc_layer is not None:
                print('===' * 10, fc_name)
                print(fc_layer)
            else:
                print(feature_net)
        except NotImplementedError:
            print('Error ' * 10)
            net = get_model(name)
            print(net)
