import argparse
import logging
import os
import random
import time
from copy import deepcopy
from test import test

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed

from autotimm.data.dataloaders import get_pytorch_train_loader, get_pytorch_val_loader
from autotimm.data.mixup import MixUpWrapper, NLLMultiLabelSmooth
from autotimm.data.smoothing import LabelSmoothing
from autotimm.models.model_zoo import get_model_list
from autotimm.models.network import get_input_size, init_network
from autotimm.optim.optimizers import get_optimizer
from autotimm.scheduler import CosineLRScheduler, ExponentialLRScheduler, LinearLRScheduler, StepLRScheduler
from autotimm.training import train_loop
from autotimm.utils.model import resum_checkpoint


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
        help='interpolation type for resizing images: bilinear, bicubic or ')
    model_names = get_model_list()
    parser.add_argument(
        '--model',
        metavar='MODEL',
        default='resnet18',
        choices=model_names,
        help='model architecture: ' + ' | '.join(model_names) +
        ' (default: resnet18)')
    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        action='store_true',
        help='use pre-trained model')
    parser.add_argument(
        '-j',
        '--workers',
        type=int,
        default=4,
        metavar='N',
        help='how many training processes to use (default: 1)')
    parser.add_argument(
        '--epochs',
        default=90,
        type=int,
        metavar='N',
        help='number of total epochs to run')
    parser.add_argument(
        '--run-epochs',
        default=-1,
        type=int,
        metavar='N',
        help='run only N epochs, used for checkpointing runs',
    )
    parser.add_argument(
        '--start-epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)')
    parser.add_argument(
        '--early-stopping-patience',
        default=-1,
        type=int,
        metavar='N',
        help='early stopping after N epochs without improving',
    )
    parser.add_argument(
        '--image-size', default=None, type=int, help='resolution of image')
    parser.add_argument(
        '-b',
        '--batch-size',
        default=256,
        type=int,
        metavar='N',
        help='mini-batch size (default: 256) per gpu')
    parser.add_argument(
        '--optimizer-batch-size',
        default=-1,
        type=int,
        metavar='N',
        help='size of a total batch size',
    )
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=0.1,
        type=float,
        metavar='LR',
        help='initial learning rate',
        dest='lr')
    parser.add_argument(
        '--end-lr',
        '--minimum learning-rate',
        default=1e-8,
        type=float,
        metavar='END-LR',
        help='initial learning rate')
    parser.add_argument(
        '--lr-schedule',
        default='step',
        type=str,
        metavar='SCHEDULE',
        choices=['step', 'linear', 'cosine', 'exponential'],
        help='Type of LR schedule: {}, {}, {} , {}'.format(
            'step', 'linear', 'cosine', 'exponential'),
    )
    parser.add_argument(
        '--auto-step',
        default=True,
        type=bool,
        help='Use auto-step lr-schedule or not')
    parser.add_argument(
        '--warmup',
        default=0,
        type=int,
        metavar='E',
        help='number of warmup epochs')
    parser.add_argument(
        '--label-smoothing',
        default=0.0,
        type=float,
        metavar='S',
        help='label smoothing')
    parser.add_argument(
        '--mixup',
        default=0.0,
        type=float,
        metavar='ALPHA',
        help='mixup alpha')
    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        choices=('sgd', 'rmsprop', 'adamw'))
    parser.add_argument(
        '--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument(
        '--wd',
        '--weight-decay',
        default=1e-4,
        type=float,
        metavar='W',
        help='weight decay (default: 1e-4)',
        dest='weight_decay')
    parser.add_argument(
        '--bn-weight-decay',
        action='store_true',
        help='use weight_decay on batch normalization',
    )
    parser.add_argument(
        '--rmsprop-alpha',
        default=0.99,
        type=float,
        help='value of alpha parameter in rmsprop optimizer (default: 0.99)',
    )
    parser.add_argument(
        '--rmsprop-eps',
        default=1e-8,
        type=float,
        help='value of eps parameter in rmsprop optimizer (default: 1e-8)',
    )
    parser.add_argument(
        '--adamw-eps',
        default=1e-8,
        type=float,
        help='value of eps parameter in adamw optimizer (default: 1e-8)',
    )
    parser.add_argument(
        '--nesterov',
        action='store_true',
        help='use nesterov momentum, (default: false)',
    )
    parser.add_argument('--use-ema', default=None, type=float, help='use EMA')
    parser.add_argument(
        '--augmentation',
        type=str,
        default=None,
        choices=[
            None, 'autoaugment', 'original-mstd0.5', 'rand-m9-n3-mstd0.5',
            'augmix-m5-w4-d2'
        ],
        help='augmentation method',
    )
    parser.add_argument(
        '--log_interval',
        default=10,
        type=int,
        metavar='N',
        help='print frequency (default: 10)')
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
        '--training-only', action='store_true', help='do not evaluate')
    parser.add_argument(
        '--no-checkpoints',
        action='store_false',
        dest='save_checkpoints',
        help='do not store any checkpoints, useful for benchmarking',
    )
    parser.add_argument(
        '--checkpoint-filename', default='checkpoint.pth.tar', type=str)
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
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument(
        '--syn-bn',
        dest='syn_bn',
        action='store_true',
        help='use pre-trained model')
    parser.add_argument(
        '--static-loss-scale',
        type=float,
        default=1,
        help='Static loss scale',
    )
    parser.add_argument(
        '--dynamic-loss-scale',
        action='store_true',
        help='Use dynamic loss scaling.  If supplied, this argument supersedes '
        + '--static-loss-scale.',
    )
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


def prepare_for_training(args):
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

    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = args.world_size * args.batch_size

        if args.optimizer_batch_size % tbs != 0:
            print(
                'Warning: simulated batch size {} is not divisible by actual batch size {}'
                .format(args.optimizer_batch_size, tbs))
        batch_size_multiplier = int(args.optimizer_batch_size / tbs)
        print('BSM: {}'.format(batch_size_multiplier))

    start_epoch = 0
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
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == 'syntetic':
        get_val_loader = get_syntetic_loader
        get_train_loader = get_syntetic_loader
    else:
        print('Bad databackend picked')
        exit(1)

    # get data loaders
    train_loader, num_class = get_train_loader(
        args.data_path,
        'train',
        image_size,
        args.batch_size,
        args.mixup > 0.0,
        interpolation=args.interpolation,
        augmentation=args.augmentation,
        start_epoch=start_epoch,
        workers=args.workers,
        # memory_format=memory_format,
    )
    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, train_loader)

    val_loader, _ = get_val_loader(
        args.data_path,
        'val',
        image_size,
        args.batch_size,
        False,
        interpolation=args.interpolation,
        workers=args.workers,
    )

    # optionally resume from a checkpoint
    if args.resume is not None:
        model_state, model_state_ema, optimizer_state, start_epoch, best_prec1 = resum_checkpoint(
            args.resume)
    else:
        model_state = None
        model_state_ema = None
        optimizer_state = None

    # model
    model = init_network(args.model, num_class, pretrained=args.pretrained)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu).to(memory_format=memory_format)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], output_device=args.gpu)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(
                model, output_device=0)
    else:
        model.cuda().to(memory_format=memory_format)

    # optionally resume from a checkpoint
    if args.resume is not None:
        model_state, model_state_ema, optimizer_state, start_epoch, best_prec1 = resum_checkpoint(
            args.resume)
    else:
        model_state = None
        model_state_ema = None
        optimizer_state = None

    # EMA
    if args.use_ema is not None:
        model_ema = deepcopy(model)
        ema = EMA(args.use_ema)
    else:
        model_ema = None
        ema = None

    # load mode state
    if model_state is not None:
        model.load_model_state(model_state)

    if (ema is not None) and (model_state_ema is not None):
        print('load ema')
        ema.load_state_dict(model_state_ema)

    # define loss function (criterion) and optimizer
    criterion = loss().cuda(args.gpu)
    # optimizer and lr_policy
    optimizer = get_optimizer(
        list(model.named_parameters()),
        args.lr,
        args=args,
        state=optimizer_state,
    )

    if args.lr_schedule == 'step':
        if args.auto_step:
            step_ratios = [0.6, 0.9]
            auto_steps = [int(ratio * args.epochs) for ratio in step_ratios]
            lr_policy = StepLRScheduler(
                optimizer=optimizer,
                base_lr=args.lr,
                steps=auto_steps,
                decay_factor=0.1,
                warmup_length=args.warmup,
                logger=logger)
        else:
            lr_policy = StepLRScheduler(
                optimizer=optimizer,
                base_lr=args.lr,
                steps=[30, 60, 80],
                decay_factor=0.1,
                warmup_length=args.warmup,
                logger=logger)
    elif args.lr_schedule == 'cosine':
        lr_policy = CosineLRScheduler(
            optimizer=optimizer,
            base_lr=args.lr,
            warmup_length=args.warmup,
            epochs=args.epochs,
            end_lr=args.end_lr,
            logger=logger)
    elif args.lr_schedule == 'linear':
        lr_policy = LinearLRScheduler(
            optimizer=optimizer,
            base_lr=args.lr,
            warmup_length=args.warmup,
            epochs=args.epochs,
            logger=logger)
    elif args.lr_schedule == 'exponential':
        lr_policy = ExponentialLRScheduler(
            optimizer=optimizer,
            base_lr=args.lr,
            warmup_length=args.warmup,
            epochs=args.epochs,
            logger=logger)

    scaler = torch.cuda.amp.GradScaler(
        init_scale=args.static_loss_scale,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=100 if args.dynamic_loss_scale else 1000000000,
        enabled=args.amp,
    )

    return (model, criterion, optimizer, lr_policy, scaler, train_loader,
            val_loader, ema, model_ema, batch_size_multiplier, start_epoch,
            num_class)


def main(args):
    global best_prec1
    best_prec1 = 0
    model, criterion, optimizer, lr_policy, scaler, train_loader, val_loader, ema, model_ema, batch_size_multiplier, \
        start_epoch, num_class = prepare_for_training(args)

    train_loop(
        model,
        criterion,
        optimizer,
        scaler,
        lr_policy,
        train_loader,
        val_loader,
        num_class=num_class,
        logger=logger,
        ema=ema,
        model_ema=model_ema,
        use_amp=args.amp,
        batch_size_multiplier=batch_size_multiplier,
        start_epoch=start_epoch,
        end_epoch=min((start_epoch + args.run_epochs), args.epochs)
        if args.run_epochs != -1 else args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        best_prec1=best_prec1,
        skip_training=args.evaluate,
        skip_validation=args.training_only,
        save_checkpoints=args.save_checkpoints and not args.evaluate,
        checkpoint_dir=args.output_dir,
        checkpoint_filename=args.checkpoint_filename,
    )
    print('Experiment ended')


if __name__ == '__main__':
    args = parse_args()
    task_name = args.data_name + '-' + args.model
    args.output_dir = os.path.join(args.output_dir, task_name)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank(
    ) == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    logger = logging.getLogger('')
    filehandler = logging.FileHandler(
        os.path.join(args.output_dir, 'summary.log'))
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    cudnn.benchmark = True
    start_time = time.time()
    main(args)

    train_end_time = time.time()
    train_time = train_end_time - start_time
    args.resume = os.path.join(args.output_dir, 'model_best.pth.tar')
    prec1 = test(args, logger)
    test_time = time.time() - train_end_time
    if not torch.distributed.is_initialized() or torch.distributed.get_rank(
    ) == 0:
        logger.info('Test Acc of Top1 is %s' % prec1)
        logger.info('Total time of train is {:7.1f} s'.format(train_time))
        logger.info('Total time of test is {:7.1f} s'.format(test_time))
