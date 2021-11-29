import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast
from copy import deepcopy
from typing import Callable, Dict, Optional, Tuple
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from autotimm.utils.metrics import AverageMeter, accuracy
from autotimm.utils.model import reduce_tensor, save_checkpoint
from autotimm.utils.time_handler import TimeoutHandler
from autotimm.models.common import EMA

class Executor:
    def __init__(
        self,
        model: nn.Module,
        loss: Optional[nn.Module],
        cuda: bool = True,
        memory_format: torch.memory_format = torch.contiguous_format,
        amp: bool = False,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        divide_loss: int = 1,
        ts_script: bool = False,
    ):
        assert not (amp and scaler is None), "Gradient Scaler is needed for AMP"

        def xform(m: nn.Module) -> nn.Module:
            if cuda:
                m = m.cuda()
            m.to(memory_format=memory_format)
            return m

        self.model = xform(model)
        if ts_script:
            self.model = torch.jit.script(self.model)
        self.ts_script = ts_script
        self.loss = xform(loss) if loss is not None else None
        self.amp = amp
        self.scaler = scaler
        self.is_distributed = False
        self.divide_loss = divide_loss
        self._fwd_bwd = None
        self._forward = None

    def distributed(self, gpu_id):
        self.is_distributed = True
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.model = DDP(self.model, device_ids=[gpu_id], output_device=gpu_id)
        torch.cuda.current_stream().wait_stream(s)

    def _fwd_bwd_fn(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        with autocast(enabled=self.amp):
            loss = self.loss(self.model(input), target)
            loss /= self.divide_loss

        self.scaler.scale(loss).backward()
        return loss

    def _forward_fn(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad(), autocast(enabled=self.amp):
            output = self.model(input)
            loss = None if self.loss is None else self.loss(output, target)

        return output if loss is None else loss, output

    def optimize(self, fn):
        return fn

    @property
    def forward_backward(self):
        if self._fwd_bwd is None:
            if self.loss is None:
                raise NotImplementedError(
                    "Loss must not be None for forward+backward step"
                )
            self._fwd_bwd = self.optimize(self._fwd_bwd_fn)
        return self._fwd_bwd

    @property
    def forward(self):
        if self._forward is None:
            self._forward = self.optimize(self._forward_fn)
        return self._forward

    def train(self):
        self.model.train()
        if self.loss is not None:
            self.loss.train()

    def eval(self):
        self.model.eval()
        if self.loss is not None:
            self.loss.eval()




class Trainer:
    def __init__(
        self,
        executor: Executor,
        optimizer: torch.optim.Optimizer,
        grad_acc_steps: int,
        ema: Optional[float] = None,
    ):
        self.executor = executor
        self.optimizer = optimizer
        self.grad_acc_steps = grad_acc_steps
        self.use_ema = False
        if ema is not None:
            self.ema_executor = deepcopy(self.executor)
            self.ema = EMA(ema, self.ema_executor.model)
            self.use_ema = True

        self.optimizer.zero_grad(set_to_none=True)
        self.steps_since_update = 0

    def train(self):
        self.executor.train()
        if self.use_ema:
            self.ema_executor.train()

    def eval(self):
        self.executor.eval()
        if self.use_ema:
            self.ema_executor.eval()

    def train_step(self, input, target, step=None):
        loss = self.executor.forward_backward(input, target)

        self.steps_since_update += 1

        if self.steps_since_update == self.grad_acc_steps:
            if self.executor.scaler is not None:
                self.executor.scaler.step(self.optimizer)
                self.executor.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps_since_update = 0

        torch.cuda.synchronize()

        if self.use_ema:
            self.ema(self.executor.model, step=step)

        return loss

    def validation_steps(self) -> Dict[str, Callable]:
        vsd: Dict[str, Callable] = {"val": self.executor.forward}
        if self.use_ema:
            vsd["val_ema"] = self.ema_executor.forward
        return vsd

    def state_dict(self) -> dict:
        res = {
            "state_dict": self.executor.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.use_ema:
            res["state_dict_ema"] = self.ema_executor.model.state_dict()

        return res



def get_train_step(model,
                   criterion,
                   optimizer,
                   scaler,
                   use_amp=False,
                   batch_size_multiplier=1,
                   top_k=1):

    def _step(input, target, optimizer_step=True):
        input_var = Variable(input)
        target_var = Variable(target)

        with autocast(enabled=use_amp):
            output = model(input_var)
            loss = criterion(output, target_var)

            loss /= batch_size_multiplier
            prec1, prec5 = accuracy(
                output.data, target, topk=(1, min(top_k, 5)))
            if torch.distributed.is_initialized():
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

        scaler.scale(loss).backward()

        if optimizer_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def train(train_loader,
          model,
          criterion,
          optimizer,
          scaler,
          lr_scheduler,
          num_class,
          logger,
          epoch,
          timeout_handler,
          ema=None,
          use_amp=False,
          batch_size_multiplier=1,
          log_interval=1):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')
    top1_m = AverageMeter('Acc@1', ':6.2f')
    top5_m = AverageMeter('Acc@5', ':6.2f')

    interrupted = False
    step = get_train_step(
        model,
        criterion,
        optimizer,
        scaler=scaler,
        use_amp=use_amp,
        batch_size_multiplier=batch_size_multiplier,
        top_k=num_class)

    model.train()
    optimizer.zero_grad()
    steps_per_epoch = len(train_loader)
    data_iter = enumerate(train_loader)
    end = time.time()
    batch_size = 1
    for i, (input, target) in data_iter:
        input = input.cuda()
        target = target.cuda()

        bs = input.size(0)
        lr_scheduler.step(epoch)
        data_time = time.time() - end

        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        loss, prec1, prec5 = step(input, target, optimizer_step=optimizer_step)
        if ema is not None:
            ema(model, epoch * steps_per_epoch + i)

        it_time = time.time() - end
        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), bs)
        top1_m.update(prec1.item(), bs)
        top5_m.update(prec5.item(), bs)

        end = time.time()
        if ((i + 1) % 20 == 0) and timeout_handler.interrupted:
            time.sleep(5)
            interrupted = True
            break
        if i == 1:
            batch_size = bs
        if (i % log_interval == 0) or (i == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                log_name = 'Train-log'
                logger.info(
                    '{0}: [epoch:{1:>2d}] [{2:>2d}/{3}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'BatchTime: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f}) '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f}) '
                    'lr: {lr:>4.6f} '.format(
                        log_name,
                        epoch + 1,
                        i,
                        steps_per_epoch,
                        data_time=data_time_m,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m,
                        lr=learning_rate))

    return interrupted, losses_m.avg, top1_m.avg / 100.0, top5_m.avg / 100.0, batch_size


def get_val_step(model, criterion, use_amp=False, top_k=1):

    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad(), autocast(enabled=use_amp):
            output = model(input_var)
            loss = criterion(output, target_var)

            prec1, prec5 = accuracy(
                output.data, target, topk=(1, min(5, top_k)))

            if torch.distributed.is_initialized():
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(val_loader,
             model,
             criterion,
             num_class,
             logger,
             logger_name,
             use_amp=False,
             log_interval=10):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')
    top1_m = AverageMeter('Acc@1', ':6.2f')
    top5_m = AverageMeter('Acc@5', ':6.2f')

    step = get_val_step(model, criterion, use_amp=use_amp, top_k=num_class)
    # switch to evaluate mode
    model.eval()
    steps_per_epoch = len(val_loader)
    end = time.time()
    data_iter = enumerate(val_loader)
    batch_size = 1
    for i, (input, target) in data_iter:
        input = input.cuda()
        target = target.cuda()

        bs = input.size(0)
        data_time = time.time() - end
        loss, prec1, prec5 = step(input, target)
        it_time = time.time() - end
        end = time.time()

        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), bs)
        top1_m.update(prec1.item(), bs)
        top5_m.update(prec5.item(), bs)

        if i == 1:
            batch_size = bs

        if (i % log_interval == 0) or (i == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                logger.info(
                    '{0}: [{1:>2d}/{2}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f}) '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        logger_name,
                        i,
                        steps_per_epoch,
                        data_time=data_time_m,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m))
    return losses_m.avg, top1_m.avg / 100.0, top5_m.avg / 100.0, batch_size


def train_loop(
    model,
    criterion,
    optimizer,
    scaler,
    lr_scheduler,
    train_loader,
    val_loader,
    num_class,
    logger,
    ema=None,
    model_ema=None,
    use_amp=False,
    batch_size_multiplier=1,
    best_prec1=0,
    start_epoch=0,
    end_epoch=0,
    early_stopping_patience=-1,
    skip_training=False,
    skip_validation=False,
    save_checkpoints=True,
    checkpoint_dir='./',
    checkpoint_filename='checkpoint.pth.tar',
):
    prec1 = -1
    use_ema = (model_ema is not None) and (ema is not None)

    if early_stopping_patience > 0:
        epochs_since_improvement = 0

    print(f'RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}')
    with TimeoutHandler() as timeout_handler:
        interrupted = False
        for epoch in range(start_epoch, end_epoch):
            if not skip_training:
                tic = time.time()
                interrupted, losses_m, top1_m, top5_m, batch_size = train(
                    train_loader,
                    model,
                    criterion,
                    optimizer,
                    scaler,
                    lr_scheduler,
                    num_class,
                    logger,
                    epoch,
                    timeout_handler,
                    ema=ema,
                    use_amp=use_amp,
                    batch_size_multiplier=batch_size_multiplier,
                    log_interval=10)

            steps_per_epoch = len(train_loader)
            throughput = int(batch_size * steps_per_epoch /
                             (time.time() - tic))
            logger.info('[Epoch %d] training: loss=%f, top1=%f, top5=%f' %
                        (epoch + 1, losses_m, top1_m, top5_m))
            logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f',
                        epoch + 1, throughput,
                        time.time() - tic)

            if not skip_validation:
                tic = time.time()
                losses_m, top1_m, top5_m, batch_size = validate(
                    val_loader,
                    model,
                    criterion,
                    num_class,
                    logger,
                    'Val-log',
                    use_amp=use_amp,
                )
                steps_per_epoch = len(val_loader)
                throughput = int(batch_size * steps_per_epoch /
                                 (time.time() - tic))
                logger.info(
                    '[Epoch %d] validation: loss=%f, top1=%f, top5=%f' %
                    (epoch + 1, losses_m, top1_m, top5_m))
                logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f',
                            epoch + 1, throughput,
                            time.time() - tic)

                if use_ema:
                    model_ema.load_state_dict({
                        k.replace('module.', ''): v
                        for k, v in ema.state_dict().items()
                    })
                    prec1 = validate(val_loader, criterion, model_ema,
                                     num_class, logger, 'Val-log')

                if prec1 > best_prec1:
                    is_best = True
                    best_prec1 = prec1
                else:
                    is_best = False
            else:
                is_best = True
                best_prec1 = 0

            if save_checkpoints and (not torch.distributed.is_initialized()
                                     or torch.distributed.get_rank() == 0):
                checkpoint_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }
                if use_ema:
                    checkpoint_state['state_dict_ema'] = ema.state_dict()

                save_checkpoint(
                    checkpoint_state,
                    is_best,
                    checkpoint_dir=checkpoint_dir,
                    filename=checkpoint_filename,
                )
            if early_stopping_patience > 0:
                if not is_best:
                    epochs_since_improvement += 1
                else:
                    epochs_since_improvement = 0
                if epochs_since_improvement >= early_stopping_patience:
                    break
            if interrupted:
                break
