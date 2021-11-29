"""Utils for auto classification estimator."""
import math
# pylint: disable=bad-whitespace,missing-function-docstring
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import datasets, transforms

from autotimm.data.autoaugment import AutoaugmentImageNetPolicy
from autotimm.data.constants import IMAGENET_DEFAULT_MEAN
from autotimm.data.dataloaders import PrefetchedWrapper, fast_collate
from autotimm.data.timm_auto_augment import _pil_interp, augment_and_mix_transform, auto_augment_transform, rand_augment_transform
from .dataset import TorchImageClassificationDataset


def get_pytorch_val_loader(data_dir,
                           batch_size,
                           num_workers,
                           input_size,
                           crop_ratio,
                           val_dataset=None,
                           start_epoch=0,
                           one_hot=False,
                           collate_fn=None,
                           _worker_init_fn=None):
    interpolation = Image.BILINEAR
    resize = int(math.ceil(input_size / crop_ratio))
    transform_test = transforms.Compose([
        transforms.Resize(resize, interpolation=interpolation),
        transforms.CenterCrop(input_size),
    ])

    if val_dataset is None:
        val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'val'), transform_test)
    else:
        assert isinstance(val_dataset, pd.DataFrame), 'DataSet Type Error'
        assert isinstance(
            val_dataset, TorchImageClassificationDataset), 'DataSet Type Error'
        val_dataset = val_dataset.to_pytorch(transform_test)

    num_classes = len(val_dataset.classes)

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=True)
    else:
        val_sampler = None

    if collate_fn is None:
        collate_fn = fast_collate

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=False,
    )
    return PrefetchedWrapper(val_loader, start_epoch, num_classes, one_hot)


def get_pytorch_train_loader(data_dir,
                             batch_size,
                             num_workers,
                             input_size,
                             crop_ratio,
                             data_augment,
                             interpolation='bilinear',
                             mean=IMAGENET_DEFAULT_MEAN,
                             train_dataset=None,
                             one_hot=False,
                             start_epoch=0,
                             collate_fn=None,
                             _worker_init_fn=None):

    interpolation_m = _pil_interp(interpolation)
    jitter_param = 0.4
    crop_ratio = crop_ratio if crop_ratio > 0 else 0.875
    transforms_list = [
        transforms.RandomResizedCrop(
            input_size, interpolation=interpolation_m),
        transforms.RandomHorizontalFlip(),
    ]

    autogluon_transforms = [
        transforms.ColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
    ]

    if data_augment:
        assert isinstance(data_augment, str)
        if isinstance(input_size, (tuple, list)):
            img_size_min = min(input_size)
        else:
            img_size_min = input_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = _pil_interp(interpolation)
        if data_augment == 'autoaugment':
            transforms_list.append(AutoaugmentImageNetPolicy())
        elif data_augment == 'autogluon':
            transforms_list.append(autogluon_transforms)
        elif data_augment.startswith('rand'):
            transforms_list.append(
                rand_augment_transform(data_augment, aa_params))
        elif data_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            transforms_list.append(
                augment_and_mix_transform(data_augment, aa_params))
        else:
            transforms_list.append(
                auto_augment_transform(data_augment, aa_params))

    transform_train = transforms.Compose(transforms_list)

    if train_dataset is None:
        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'), transform_train)
    else:
        assert isinstance(train_dataset, pd.DataFrame), 'DataSet Type Error'
        # assert isinstance(
        #     train_dataset,
        #     TorchImageClassificationDataset), "DataSet Type Error"
        train_dataset = train_dataset.to_pytorch(transform_train)

    num_classes = len(train_dataset.classes)
    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True)
    else:
        train_sampler = None

    if collate_fn is None:
        collate_fn = fast_collate

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=False,
    )
    return PrefetchedWrapper(train_loader, start_epoch, num_classes, one_hot)


def get_data_loader(data_dir,
                    batch_size,
                    num_workers,
                    input_size,
                    crop_ratio,
                    data_augment,
                    train_dataset=None,
                    val_dataset=None):
    """AutoPytorch ImageClassification data loaders
    Parameters:
    -----------
    data_dir:
        data_dir
    batch_size:
        batch_szie
    num_workers:
        4
    input_size:
         224
    crop_ratio:
        0.875
    data_augment:
        None
    train_dataset:
        TorchImageClassificationDataset
    val_dataset:
        TorchImageClassificationDataset
    """
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    jitter_param = 0.4
    crop_ratio = crop_ratio if crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.ToTensor(), normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(), normalize
    ])

    if train_dataset is None:
        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'), transform_train)
    else:
        assert isinstance(train_dataset, pd.DataFrame), 'DataSet Type Error'
        # assert isinstance(
        #     train_dataset,
        #     TorchImageClassificationDataset), "DataSet Type Error"
        train_dataset = train_dataset.to_pytorch(transform_train)

    if val_dataset is None:
        val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'val'), transform_test)
    else:
        assert isinstance(val_dataset, pd.DataFrame), 'DataSet Type Error'
        assert isinstance(
            val_dataset, TorchImageClassificationDataset), 'DataSet Type Error'
        val_dataset = val_dataset.to_pytorch(transform_test)

    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return train_data, val_data
