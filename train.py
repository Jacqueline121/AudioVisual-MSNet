import torch
import torch.nn.functional as F
import time
import os

from utils import AverageMeter
from models.sum_losses import cross_entropy_loss


def train_epoch(epoch, data_loader, model, optimizer, opt):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_out = {'global':AverageMeter(), 'sum':AverageMeter()}

    end_time = time.time()
    for i, (data, valid) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = data['gt'].cuda().float()
            valid['sum'] = valid['sum'].cuda()

        inputs = data['rgb']
        valid['sum'] = valid['sum'].float()
        outputs = model(inputs, data['audio'])
        loss = cross_entropy_loss(outputs, targets)

        loss.backward()
        optimizer['sum'].step()
        optimizer['sum'].zero_grad()
        optimizer['sound'].step()
        optimizer['sound'].zero_grad()
        optimizer['fusion'].step()
        optimizer['fusion'].zero_grad()
        losses_out['global'].update(loss.data, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses_out['global']))

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer_global': optimizer['global'].state_dict(),
            'optimizer_sum': optimizer['sum'].state_dict(),
            'optimizer_sound': optimizer['sound'].state_dict(),
            'optimizer_fusion': optimizer['fusion'].state_dict()
        }
        torch.save(states, save_file_path)

    return opt
