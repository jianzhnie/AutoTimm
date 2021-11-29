import argparse
import os
import time
import yaml
import shutil
import warnings
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils
from torchvision import datasets, models, transforms
import autogluon.core as ag
from autotorch.utils.model import save_checkpoint, reduce_tensor, adjust_learning_rate, load_checkpoint
from autotorch.utils.metrics import AverageMeter, accuracy
from autotorch.models.model_zoo import get_model_list
from autotorch.models.network import get_input_size, init_network
from autotorch.data.dataloaders import get_pytorch_train_loader, get_pytorch_val_loader


model_names = get_model_list()


def parse_args():
    parser = argparse.ArgumentParser(description='Model-based Asynchronous HPO')
    parser.add_argument('--data_name', default="", type=str, help='dataset name')
    parser.add_argument('--data_path', default="", type=str, help='path to dataset')
    parser.add_argument('--model', metavar='MODEL', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                        help='how many training processes to use (default: 1)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--image-size', default=None, type=int, help="resolution of image")
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-schedule', default="step", type=str, metavar="SCHEDULE",
                        choices=["step", "linear", "cosine"],
                        help="Type of LR schedule: {}, {}, {}".format("step", "linear", "cosine"),)
    parser.add_argument('--warmup', default=0, type=int, metavar="E", help="number of warmup epochs")
    parser.add_argument('--label-smoothing', default=0.0, type=float, metavar="S", help="label smoothing")
    parser.add_argument('--mixup', default=0.0, type=float, metavar="ALPHA", help="mixup alpha")
    parser.add_argument('--optimizer', default="sgd", type=str, choices=("sgd", "rmsprop"))
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--augmentation', type=str, default=None, choices=[None, "autoaugment", "original-mstd0.5", "rand-m9-n3-mstd0.5", "augmix-m5-w4-d2"], 
                        help="augmentation method",)
    parser.add_argument('--log_interval', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--output-dir', default="/home/yiran.wu/work_dirs/pytorch_model_benchmark", type=str,
                        help='output directory for model and log')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')

    args = parser.parse_args()
    return args


torch.backends.cudnn.benchmark = True


def train_loop():
    opt = parse_args()
    task_name = opt.data_name + '-' + opt.model
    opt.output_dir = os.path.join(opt.output_dir, task_name)
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    _logger = logging.getLogger('')
    filehandler = logging.FileHandler(os.path.join(opt.output_dir, 'summary.log'))
    streamhandler = logging.StreamHandler()
    _logger.setLevel(logging.INFO)
    _logger.addHandler(filehandler)
    _logger.addHandler(streamhandler)

    ngpus_per_node = torch.cuda.device_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_size = get_input_size(opt.model)
    # dataloaders, num_class = load_data(opt.data_path, input_size, batch_size, ngpus_per_node)

    train_loader, num_classes = get_pytorch_train_loader(opt.data_path, "train", input_size, opt.batch_size, augmentation = opt.augmentation)
    valid_loader, _ = get_pytorch_val_loader(opt.data_path, "val", input_size, opt.batch_size)
    test_batch_size = int(opt.batch_size/ max(ngpus_per_node, 1))
    test_loader, _ = get_pytorch_val_loader(opt.data_path, "test", input_size, test_batch_size)

    # model ddp
    opt.distributed = False
    if 'WORLD_SIZE' in os.environ:
        opt.distributed = int(os.environ['WORLD_SIZE']) > 1
        opt.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        opt.local_rank = 0

    opt.device = 'cuda:0'
    opt.world_size = 1
    opt.rank = 0  # global rank

    if opt.distributed:
        opt.device = 'cuda:%d' % opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        opt.world_size = torch.distributed.get_world_size()
        opt.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (opt.rank, opt.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert opt.rank >= 0

    # model
    model = init_network(opt.model, num_classes, pretrained=opt.pretrained)
    # move model to GPU, enable channels last layout if set
    model.cuda()
    
    # setup distributed training
    if opt.distributed:
        if opt.local_rank == 0:
            _logger.info("Using native Torch DistributedDataParallel.")
        model = NativeDDP(model, device_ids=[opt.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP
    criterion = nn.CrossEntropyLoss().cuda()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

    best_acc1 = 0

    # Training
    def train(epoch, loader, num_classes):
        batch_time_m = AverageMeter('Time', ':6.3f')
        data_time_m = AverageMeter('Data', ':6.3f')
        losses_m = AverageMeter('Loss', ':.4e')
        top1_m = AverageMeter('Acc@1', ':6.2f')
        top5_m = AverageMeter('Acc@5', ':6.2f')

        # loader = dataloaders['train']

        lr = adjust_learning_rate(optimizer, epoch, opt)

        model.train()

        end = time.time()
        last_idx = len(loader) - 1
        num_updates = epoch * len(loader)

        for batch_idx, (inputs, targets) in enumerate(loader):
            last_batch = batch_idx == last_idx
            data_time_m.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets) 
            acc1, acc5 = accuracy(outputs, targets, topk=(1, min(num_classes, 5)))
            top1_m.update(acc1.item(), outputs.size(0))
            top5_m.update(acc5.item(), outputs.size(0))
            loss.backward()
            optimizer.step()
            num_updates += 1
            batch_time_m.update(time.time() - end)
            if last_batch or batch_idx % opt.log_interval == 0:
                if opt.distributed:
                    reduced_loss = reduce_tensor(loss.data)
                    acc1 = reduce_tensor(acc1)
                    acc5 = reduce_tensor(acc5)
                    losses_m.update(reduced_loss.item(), inputs.size(0))
                else:
                    losses_m.update(loss.item(), inputs.size(0))
                if opt.local_rank == 0:
                    _logger.info(
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            epoch,
                            batch_idx, len(loader),
                            100. * batch_idx / last_idx,
                            loss=losses_m,
                            top1=top1_m, 
                            top5=top5_m,
                            batch_time=batch_time_m,
                            rate=inputs.size(0) * opt.world_size / batch_time_m.val,
                            rate_avg=inputs.size(0) * opt.world_size / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m))

    def val(loader):
        batch_time_m = AverageMeter('Time', ':6.3f')
        data_time_m = AverageMeter('Data', ':6.3f')
        losses_m = AverageMeter('Loss', ':.4e')
        top1_m = AverageMeter('Acc@1', ':6.2f')
        top5_m = AverageMeter('Acc@5', ':6.2f')

        model.eval()
        end = time.time()
        last_idx = len(loader) - 1
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                last_batch = batch_idx == last_idx
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, min(num_classes, 5)))
                torch.cuda.synchronize()
                if opt.distributed:
                    reduced_loss = reduce_tensor(loss.data)
                    losses_m.update(reduced_loss.item(), inputs.size(0))
                    acc1 = reduce_tensor(acc1)
                    acc5 = reduce_tensor(acc5)
                else:
                    losses_m.update(loss.item(), inputs.size(0))
                top1_m.update(acc1.item(), outputs.size(0))
                top5_m.update(acc5.item(), outputs.size(0))
                batch_time_m.update(time.time() - end)
                end = time.time()
                if opt.local_rank == 0 and (last_batch or batch_idx % opt.log_interval == 0):
                    log_name = 'Val-log'
                    _logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx, batch_time=batch_time_m,
                            loss=losses_m, top1=top1_m, top5=top5_m))
        val_acc1 = top1_m.avg
        return val_acc1


    def test(loader):
        batch_time_m = AverageMeter('Time', ':6.3f')
        data_time_m = AverageMeter('Data', ':6.3f')
        losses_m = AverageMeter('Loss', ':.4e')
        top1_m = AverageMeter('Acc@1', ':6.2f')
        top5_m = AverageMeter('Acc@5', ':6.2f')

        model.eval()
        end = time.time()
        last_idx = len(loader) - 1
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                last_batch = batch_idx == last_idx
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, min(num_classes, 5)))
                torch.cuda.synchronize()

                if opt.distributed:
                    reduced_loss = reduce_tensor(loss.data)
                    losses_m.update(reduced_loss.item(), inputs.size(0))
                    acc1 = reduce_tensor(acc1)
                    acc5 = reduce_tensor(acc5)
                else:
                    losses_m.update(loss.item(), inputs.size(0))
                top1_m.update(acc1.item(), outputs.size(0))
                top5_m.update(acc5.item(), outputs.size(0))

                batch_time_m.update(time.time() - end)
                end = time.time()
                if opt.local_rank == 0 and (last_batch or batch_idx % opt.log_interval == 0):
                    log_name = 'Test-log'
                    _logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx, batch_time=batch_time_m,
                            loss=losses_m, top1=top1_m, top5=top5_m))
        _logger.info("The top 1 test accuracy of best model is {:>7.4f}".format(top1_m.avg))


    for epoch in range(0, opt.epochs):
        train(epoch, train_loader, num_classes)
        val_acc1 = val(valid_loader)
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed
                and opt.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint_dir = opt.output_dir)
    
    print("Training and validation finished!\n")

    print("Testing start now!\n")

    # print(list(model.modules()))
    model = init_network(opt.model, num_classes, pretrained=opt.pretrained)
    # move model to GPU, enable channels last layout if set
    model.cuda()
    best_ckpt_path = os.path.join(opt.output_dir, 'model_best.pth.tar')
    load_checkpoint(model, best_ckpt_path)
    test(test_loader)


# @ag.args(
#     learning_rate=ag.space.Real(lower=1e-6, upper=1, log=True),
#     momentum=ag.space.Real(lower=0.88, upper=0.9),
#     batch_size=ag.space.Int(lower=128, upper=256),
#     epochs=10,
# )
# def train_finetune(args, reporter):
#     return train_loop(args, reporter)


if __name__ == '__main__':
    # myscheduler = ag.scheduler.FIFOScheduler(train_finetune,
    #                                         resource={'num_cpus': 32, 'num_gpus': 4},
    #                                         checkpoint='checkpoint',
    #                                         num_trials=2,
    #                                         time_attr='epoch',
    #                                         reward_attr="accuracy")

    # # Run experiment
    # myscheduler.run()
    # myscheduler.join_jobs()
    # myscheduler.get_training_curves(plot=True, use_legend=False)
    # print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
    #                                                             myscheduler.get_best_reward()))
    train_loop()
