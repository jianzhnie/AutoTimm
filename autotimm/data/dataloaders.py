import os

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from timm.data.mixup import FastCollateMixup

from .autoaugment import AutoaugmentImageNetPolicy
from .constants import IMAGENET_DEFAULT_MEAN
from .timm_auto_augment import _pil_interp, augment_and_mix_transform, auto_augment_transform, rand_augment_transform


def fast_collate(batch):
    assert isinstance(batch[0], tuple)
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros(
        (len(imgs), 3, h, w),
        dtype=torch.uint8).contiguous(memory_format=torch.contiguous_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array.copy())

    return tensor, targets


def expand(num_classes, dtype, tensor):
    e = torch.zeros(
        tensor.size(0), num_classes, dtype=dtype, device=torch.device('cuda'))
    e = e.scatter(1, tensor.unsqueeze(1), 1.0)
    return e


class PrefetchedWrapper(object):

    def prefetched_loader(loader, num_classes, one_hot):
        mean = (
            torch.tensor([0.485 * 255, 0.456 * 255,
                          0.406 * 255]).cuda().view(1, 3, 1, 1))
        std = (
            torch.tensor([0.229 * 255, 0.224 * 255,
                          0.225 * 255]).cuda().view(1, 3, 1, 1))

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float()
                if one_hot:
                    next_target = expand(num_classes, torch.float, next_target)

                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, start_epoch, num_classes, one_hot):
        self.dataloader = dataloader
        self.epoch = start_epoch
        self.one_hot = one_hot
        self.num_classes = num_classes

    def __iter__(self):
        if self.dataloader.sampler is not None and isinstance(
                self.dataloader.sampler,
                torch.utils.data.distributed.DistributedSampler):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader,
                                                   self.num_classes,
                                                   self.one_hot)

    def __len__(self):
        return len(self.dataloader)

    @property
    def mixup_enabled(self):
        if isinstance(self.dataloader.collate_fn, FastCollateMixup):
            return self.dataloader.collate_fn.mixup_enabled
        else:
            return False

    @mixup_enabled.setter
    def mixup_enabled(self, x):
        if isinstance(self.dataloader.collate_fn, FastCollateMixup):
            self.dataloader.collate_fn.mixup_enabled = x


def get_pytorch_train_loader(
    data_path,
    split_dir,
    image_size,
    batch_size,
    one_hot=False,
    interpolation='bilinear',
    augmentation=None,
    mean=IMAGENET_DEFAULT_MEAN,
    start_epoch=0,
    workers=5,
    collate_fn=None,
    _worker_init_fn=None,
):
    interpolation_m = _pil_interp(interpolation)
    traindir = os.path.join(data_path, split_dir)
    transforms_list = [
        transforms.RandomResizedCrop(
            image_size, interpolation=interpolation_m),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ]
    jitter_param = 0.4
    autogluon_transforms = [
        transforms.ColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
    ]

    if augmentation:
        assert isinstance(augmentation, str)
        if isinstance(image_size, (tuple, list)):
            img_size_min = min(image_size)
        else:
            img_size_min = image_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = _pil_interp(interpolation)
        if augmentation == 'autoaugment':
            transforms_list.append(AutoaugmentImageNetPolicy())
        elif augmentation == 'autogluon':
            transforms_list.append(autogluon_transforms)
        elif augmentation.startswith('rand'):
            transforms_list.append(
                rand_augment_transform(augmentation, aa_params))
        elif augmentation.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            transforms_list.append(
                augment_and_mix_transform(augmentation, aa_params))
        else:
            transforms_list.append(
                auto_augment_transform(augmentation, aa_params))

    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose(transforms_list))

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
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=True)

    return (PrefetchedWrapper(
        train_loader,
        start_epoch=start_epoch,
        num_classes=num_classes,
        one_hot=one_hot), num_classes)


def get_pytorch_val_loader(data_path,
                           split_dir,
                           image_size,
                           batch_size,
                           one_hot=False,
                           interpolation='bilinear',
                           workers=5,
                           crop_padding=32,
                           collate_fn=None,
                           _worker_init_fn=None):
    interpolation = {
        'bicubic': Image.BICUBIC,
        'bilinear': Image.BILINEAR
    }[interpolation]
    valdir = os.path.join(data_path, split_dir)
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(
                image_size + crop_padding, interpolation=interpolation),
            transforms.CenterCrop(image_size),
        ]),
    )
    num_classes = len(val_dataset.classes)

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False)
    else:
        val_sampler = None

    if collate_fn is None:
        collate_fn = fast_collate

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=True,
    )

    return PrefetchedWrapper(
        val_loader, start_epoch=0, num_classes=num_classes,
        one_hot=one_hot), num_classes


class SynteticDataLoader(object):

    def __init__(
        self,
        batch_size,
        num_classes,
        num_channels,
        height,
        width,
        one_hot,
        memory_format=torch.contiguous_format,
    ):
        input_data = (
            torch.randn(batch_size, num_channels, height, width).contiguous(
                memory_format=memory_format).cuda().normal_(0, 1.0))
        if one_hot:
            input_target = torch.empty(batch_size, num_classes).cuda()
            input_target[:, 0] = 1.0
        else:
            input_target = torch.randint(0, num_classes, (batch_size, ))
        input_target = input_target.cuda()

        self.input_data = input_data
        self.input_target = input_target

    def __iter__(self):
        while True:
            yield self.input_data, self.input_target


def get_syntetic_loader(
    data_path,
    image_size,
    batch_size,
    num_classes,
    one_hot,
    interpolation=None,
    augmentation=None,
    start_epoch=0,
    workers=None,
    _worker_init_fn=None,
    memory_format=torch.contiguous_format,
):
    return (
        SynteticDataLoader(
            batch_size,
            num_classes,
            3,
            image_size,
            image_size,
            one_hot,
            memory_format=memory_format,
        ),
        -1,
    )
