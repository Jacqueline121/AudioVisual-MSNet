import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (Compose, Normalize, Scale,
                                RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop
from target_transforms import Label, VideoID
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test_sum


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = False  # type: bool

    opt = parse_opts()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_devices
    opt.result_path = os.path.join(opt.root_path, opt.result_path)
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    print(opt.result_path)

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)

    torch.manual_seed(opt.manual_seed)

    if opt.audiovisual:
        opt.learning_rate_global = opt.learning_rate / 100
        opt.learning_rate_sum = opt.learning_rate / 100
        opt.learning_rate_sound = opt.learning_rate / 10
        opt.learning_rate_fusion = opt.learning_rate
    else:
        opt.learning_rate_global = opt.learning_rate / 10
        opt.learning_rate_sum = opt.learning_rate / 100
        opt.learning_rate_sound = 0
        opt.learning_rate_fusion = 0

    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    model, parameters = generate_model(opt)
    print(model)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:

        if not opt.no_hflip:
            spatial_transform = Compose([
                Scale([opt.sample_size,opt.sample_size]),
                RandomHorizontalFlip(),
                ToTensor(opt.norm_value), norm_method
            ])
        else:
            spatial_transform = Compose([
                Scale([opt.sample_size, opt.sample_size]),
                ToTensor(opt.norm_value), norm_method
            ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = Label()

        opt.dataset = 'movie50'
        training_data_movie50 = get_training_set(opt, spatial_transform, temporal_transform, target_transform)

        training_data = torch.utils.data.ConcatDataset([training_data_movie50])

        opt.batch_sizes = {'global': opt.effective_batch_size,
                           'sum': opt.effective_batch_size}

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            drop_last=True,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'), ['epoch', 'loss', 'loss_sum', 'sum_cross', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'), ['epoch', 'batch', 'iter', 'loss', 'lr'])

        spatial_transform = Compose([
            Scale([opt.sample_size, opt.sample_size]),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = VideoID()

        opt.dataset = 'movie50'
        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening

        optimizer = {'global':[], 'sum':[], 'sound':[], 'fusion':[]}
        optimizer['global'] = optim.SGD(
            parameters['global'],
            lr=opt.learning_rate_global,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)

        optimizer['sum'] = optim.SGD(
            parameters['sum'],
            lr=opt.learning_rate_sum,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)

        optimizer['sound'] = optim.SGD(
            parameters['sound'],
            lr=opt.learning_rate_sound,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)

        optimizer['fusion'] = optim.SGD(
            parameters['fusion'],
            lr=opt.learning_rate_fusion,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)


        scheduler = {'global': [], 'sum': [], 'sound': [], 'fusion': []}
        scheduler['global'] = lr_scheduler.MultiStepLR(
            optimizer['global'], [int(0.5*opt.n_epochs)-1, int(0.75 * opt.n_epochs)-1])

        scheduler['sum'] = lr_scheduler.MultiStepLR(
            optimizer['sum'], [int(0.5*opt.n_epochs)-1, int(0.75 * opt.n_epochs)-1])

        scheduler['sound'] = lr_scheduler.MultiStepLR(
            optimizer['sound'], [int(0.5*opt.n_epochs)-1, int(0.75 * opt.n_epochs)-1])

        scheduler['fusion'] = lr_scheduler.MultiStepLR(
            optimizer['fusion'], [int(0.5*opt.n_epochs)-1, int(0.75 * opt.n_epochs)-1])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer['global'].load_state_dict(checkpoint['optimizer_global'])
            optimizer['sum'].load_state_dict(checkpoint['optimizer_sum'])
            optimizer['sound'].load_state_dict(checkpoint['optimizer_sound'])
            optimizer['fusion'].load_state_dict(checkpoint['optimizer_fusion'])

    print('running...')
    opt.aveGrad = {'global': 0, 'sum': 0}
    loss = nn.BCELoss()
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            opt = train_epoch(i, opt.n_epochs, test_loader, model, optimizer, opt,
                        train_logger, train_batch_logger, loss)
            test_sum.test(test_loader, model, opt)
        if not opt.no_train:
            scheduler['global'].step()
            scheduler['sum'].step()
            scheduler['sound'].step()
            scheduler['fusion'].step()

    if opt.test:
        spatial_transform = Compose([
            Scale([opt.sample_size, opt.sample_size]),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = VideoID()

        opt.dataset = 'movie50'
        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test_sum.test(test_loader, model, opt)
