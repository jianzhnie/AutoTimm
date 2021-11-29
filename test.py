import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed

from autotimm.data.dataloaders import get_pytorch_val_loader, get_syntetic_loader
from autotimm.data.mixup import NLLMultiLabelSmooth
from autotimm.data.smoothing import LabelSmoothing
from autotimm.models.model_zoo import get_model_list
from autotimm.models.network import get_input_size, init_network
from autotimm.utils.model import test_load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model-based Asynchronous HPO')
    parser.add_argument(
        '--data_name', default='', type=str, help='dataset name')
    parser.add_argument(
        '--data_path', default='', type=str, help='path to dataset')
    parser.add_argument('--data-backend', metavar='BACKEND', default='pytorch')
    parser.add_argument(
        '--interpolation',
        metavar='INTERPOLATION',
        default='bilinear',
        help='interpolation type for resizing images: bilinear, bicubic')
    model_names = get_model_list()
    parser.add_argument(
        '--model',
        metavar='MODEL',
        default='resnet18',
        choices=model_names,
        help='model architecture: ' + ' | '.join(model_names) +
        ' (default: resnet18)')
    parser.add_argument(
        '-j',
        '--workers',
        type=int,
        default=4,
        metavar='N',
        help='how many training processes to use (default: 1)')
    parser.add_argument(
        '--image-size', default=None, type=int, help='resolution of image')
    parser.add_argument(
        '-b',
        '--batch-size',
        default=256,
        type=int,
        metavar='N',
        help='mini-batch size (default: 256), this is the total '
        'batch size of all GPUs on the current node when '
        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument(
        '--resume',
        default=None,
        type=str,
        metavar='PATH',
        help='path to latest checkpoint (default: none)')
    parser.add_argument(
        '--evaluate',
        dest='evaluate',
        action='store_true',
        help='evaluate model on validation set')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        metavar='S',
        help='random seed (default: 42)')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument(
        '--apex-amp',
        action='store_true',
        default=False,
        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument(
        '--native-amp',
        action='store_true',
        default=False,
        help='Use Native Torch AMP mixed precision')
    parser.add_argument(
        '--static-loss-scale', type=float, default=1, help='Static loss scale')
    parser.add_argument(
        '--mixup',
        default=0.0,
        type=float,
        metavar='ALPHA',
        help='mixup alpha')
    parser.add_argument(
        '--label-smoothing',
        default=0.0,
        type=float,
        metavar='S',
        help='label smoothing')

    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument(
        '--world-size',
        default=-1,
        type=int,
        help='number of nodes for distributed training')
    parser.add_argument(
        '--rank',
        default=-1,
        type=int,
        help='node rank for distributed training')
    parser.add_argument(
        '--memory-format',
        type=str,
        default='nchw',
        choices=['nchw', 'nhwc'],
        help='memory layout, nchw or nhwc',
    )
    parser.add_argument(
        '--output-dir',
        default='/home/yiran.wu/work_dirs/pytorch_model_benchmark',
        type=str,
        help='output directory for model and log')
    args = parser.parse_args()
    return args


def prepare_for_test(args):
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = 0

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        if not torch.distributed.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')
            args.world_size = torch.distributed.get_world_size()

    if args.seed is not None:
        print('Using seed = {}'.format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)

    else:

        def _worker_init_fn(id):
            pass

    if args.static_loss_scale != 1.0:
        if not args.amp:
            print(
                'Warning: if --amp is not used, static_loss_scale will be ignored.'
            )

    # set the image_size
    image_size = (
        args.image_size
        if args.image_size is not None else get_input_size(args.model))
    memory_format = (
        torch.channels_last
        if args.memory_format == 'nhwc' else torch.contiguous_format)

    # Creat train losses
    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0:
        loss = NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        loss = LabelSmoothing(args.label_smoothing)

    # Create data loaders
    if args.data_backend == 'pytorch':
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == 'syntetic':
        get_val_loader = get_syntetic_loader
    else:
        print('Bad databackend picked')
        exit(1)

    test_loader, num_class = get_val_loader(
        args.data_path,
        'test',
        image_size,
        args.batch_size,
        False,
        interpolation=args.interpolation,
        workers=args.workers,
        # memory_format=memory_format,
    )

    # model
    model = init_network(args.model, num_class, pretrained=False)

    if args.distributed:
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu).to(memory_format=memory_format)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu)
    else:
        model.cuda().to(memory_format=memory_format)

    # optionally resume from a checkpoint
    if args.resume is not None:
        model_state, model_state_ema, optimizer_state = test_load_checkpoint(
            args)
    else:
        model_state = None
        model_state_ema = None

    # EMA
    model_ema = None
    ema = None

    # load mode state
    if model_state is not None:
        print('load model checkpoint')
        model.load_state_dict(model_state, strict=False)

    if (ema is not None) and (model_state_ema is not None):
        print('load ema')
        ema.load_state_dict(model_state_ema)

    # define loss function (criterion) and optimizer
    criterion = loss().cuda(args.gpu)

    return (model, criterion, test_loader, ema, model_ema, num_class)


def test(args, logger):
    model, criterion, test_loader, ema, model_ema, num_class = prepare_for_test(
        args)
    use_ema = (model_ema is not None) and (ema is not None)
    prec1 = validate(
        test_loader,
        model,
        criterion,
        num_class,
        logger,
        'Test-log',
        use_amp=args.amp)
    if use_ema:
        model_ema.load_state_dict(
            {k.replace('module.', ''): v
             for k, v in ema.state_dict().items()})
        prec1 = validate(test_loader, model, criterion, num_class, logger,
                         'Test-log')
    return prec1


if __name__ == '__main__':
    args = parse_args()
    logger = logging.getLogger('')
    filehandler = logging.FileHandler(
        os.path.join(args.output_dir, 'summary.log'))
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    cudnn.benchmark = True
    prec1 = test(args, logger)
    logger.info('**' * 100)
    logger.info('Test Acc of Top1 is %s' % prec1)
