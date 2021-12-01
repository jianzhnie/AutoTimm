'''
Author: jianzhnie
Date: 2021-09-30 15:32:25
LastEditTime: 2021-12-01 17:55:49
LastEditors: jianzhnie
Description:

'''
import timm

timm_models = timm.list_models(pretrained=True)

__all__ = ['get_model', 'get_model_list']


def get_model(name, **kwargs):
    """Returns a pre-defined model by name
    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Returns
    -------
    Module:
        The model.
    """

    name = name.lower()
    if name in timm_models:
        net = timm.create_model(name, **kwargs)
    else:
        raise ValueError('%s\n\t%s' %
                         (str(name), '\n\t'.join(sorted(timm_models))))
    return net


def get_model_list():
    """Get the entire list of model names in model_zoo.
    Returns
    -------
    list of str
        Entire list of model names in model_zoo.
    """
    return list(timm_models)  # + list(torchvison_models)


if __name__ == '__main__':
    models = get_model_list()
    print(models)
    net = get_model('efficientnet_b1', pretrained=False)
    print(net)
