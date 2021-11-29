import os
import time
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from autotorch.models.network import init_network, get_input_size
from autotorch.auto.data.dataloader import get_pytorch_train_loader, get_pytorch_val_loader
from autotorch.utils.metrics import AverageMeter, accuracy, ProgressMeter
from autotorch.utils.model import reduce_tensor
from autotorch.auto.estimators.base_estimator import set_default
from autotorch.proxydata.default import ImageClassificationCfg
from autotorch.proxydata.sampler import read_entropy_file, get_proxy_data_random, get_proxy_data_log_entropy_histogram


@set_default(ImageClassificationCfg())
class ProxyModel():
    def __init__(self, config=None, logger=None, name=None):
        self._logger = logger if logger is not None else logging.getLogger(
            name)
        self._logger.setLevel(logging.INFO)
        self._cfg = self._default_cfg

    def fit(self, train_data, val_data=None):
        torch.backends.cudnn.benchmark = True
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # init network
        classes = train_data.classes
        num_class = len(classes)
        model_name = self._cfg.img_cls.model_name.lower()
        self.input_size = get_input_size(model_name)
        use_pretrained = self._cfg.img_cls.use_pretrained
        self.net = init_network(model_name=model_name,
                                num_class=num_class,
                                pretrained=use_pretrained)
        self.net = self.net.to(self.device)

        # init loss function
        self.criterion = nn.CrossEntropyLoss()
        # init optimizer
        self.optimizer = optim.SGD(params=self.net.parameters(),
                                   lr=self._cfg.train.base_lr,
                                   momentum=self._cfg.train.momentum,
                                   weight_decay=self._cfg.train.weight_decay,
                                   nesterov=self._cfg.train.nesterov)

        train_loader = get_pytorch_train_loader(
            data_dir=self._cfg.train.data_dir,
            batch_size=self._cfg.train.batch_size,
            num_workers=self._cfg.train.num_workers,
            input_size=self.input_size,
            crop_ratio=self._cfg.train.crop_ratio,
            data_augment=self._cfg.train.data_augment,
            train_dataset=train_data)

        val_loader = get_pytorch_val_loader(
            data_dir=self._cfg.train.data_dir,
            batch_size=self._cfg.valid.batch_size,
            num_workers=self._cfg.valid.num_workers,
            input_size=self.input_size,
            crop_ratio=self._cfg.train.crop_ratio,
            val_dataset=val_data)

        for epoch in range(0, self._cfg.train.epochs):
            # train for one epoch
            self.train(train_loader, self.net, self.criterion, self.optimizer,
                       epoch, num_class)
            # evaluate on validation set
            if val_loader is not None:
                acc1 = self.validate(val_loader, self.net, self.criterion,
                                     num_class)
                print("valid acc : %f" % acc1)

    def train(self, train_loader, model, criterion, optimizer, epoch,
              num_class):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(train_loader),
                                 [batch_time, data_time, losses, top1, top5],
                                 prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()
        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            images = images.to(self.device)
            target = target.to(self.device)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, num_class)))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if self.local_rank == 0:
            if i % self._cfg.train.log_interval == 0:
                progress.display(i)

    def validate(self, val_loader, model, criterion, num_class):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(val_loader),
                                 [batch_time, losses, top1, top5],
                                 prefix='Test: ')

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images = images.to(self.device)
                target = target.to(self.device)
                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output,
                                      target,
                                      topk=(1, min(5, num_class)))

                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self._cfg.valid.log_interval == 0:
                    progress.display(i)

            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5))
        self.net = model
        return top1.avg

    def generate_proxy_data(self,
                            train_data,
                            sampling_portion=0.2,
                            sampler_type='histogram',
                            output_dir='',
                            entropy_file_name='entropy_list.txt',
                            logging=None):
        def get_entropy(data):
            entropy = F.softmax(data, dim=1) * F.log_softmax(data + 1e-6,
                                                             dim=1)
            entropy = -1.0 * entropy.sum(dim=1)
            return entropy

        batch_size = self._cfg.test.batch_size
        data_loader = get_pytorch_val_loader(
            data_dir=self._cfg.train.data_dir,
            batch_size=batch_size,
            num_workers=self._cfg.test.num_workers,
            input_size=self.input_size,
            crop_ratio=self._cfg.train.crop_ratio,
            val_dataset=train_data)

        entropy_list = []  # result
        label_list = []

        self.net.eval()
        steps_per_epoch = len(data_loader)
        with torch.no_grad():
            for i, (input, target) in enumerate(data_loader):
                input = input.to(self.device)
                target = target.to(self.device)
                output = self.net(input)  # model must be declared
                ent = get_entropy(output.data)  # entropy extraction

                if i % self._cfg.valid.log_interval == 0 or (
                        i == steps_per_epoch - 1):
                    print("===> generate entropy value batch: %d " % i)
                # generate entropy file
                entropy_list.extend(ent.tolist())
                label_list.extend(target.data.tolist())

        # write the entropy file
        print("===> Finished generate entropy_list")
        entropy_path = os.path.join(output_dir, entropy_file_name)
        with open(entropy_path, 'w') as f:
            for idx in range(len(entropy_list)):
                f.write('%d %f %d\n' %
                        (idx, entropy_list[idx], label_list[idx]))

        # generate proxy dataset
        index, entropy, label = read_entropy_file(entropy_path)

        print("===> Write the entropy_list to csv ")
        if sampler_type == 'random':
            indices = get_proxy_data_random(entropy, sampling_portion, logging)
        elif sampler_type == 'histogram':
            indices = get_proxy_data_log_entropy_histogram(entropy,
                                                           sampling_portion,
                                                           histogram_type=1,
                                                           dataset='cifar10',
                                                           logging=logging)
        else:
            raise NotImplementedError

        proxy_data = train_data.iloc[indices, :]
        saved_path = os.path.join(output_dir, "proxy_data.csv")
        proxy_data.to_csv(saved_path, index=None)

        # releases all unoccupied cached memory
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        else:
            raise Exception
        return proxy_data


if __name__ == '__main__':
    from autotorch.auto import ImagePredictor
    train_dataset, val_dataset, test_dataset = ImagePredictor.Dataset.from_folders(
        '/data/AutoML_compete/food-101/split/')

    proxmodel = ProxyModel()
    proxmodel.fit(train_dataset, val_dataset)
    proxy_data = proxmodel.generate_proxy_data(
        train_data=train_dataset,
        output_dir='/data/AutoML_compete/food-101/split/')