import torch
import torch.nn.functional as F
import time
import os
import sys
import numpy as np
from imageio import imwrite

from utils import AverageMeter

def normalize_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    data_norm = np.clip((data - data_min) *
                        (255.0 / (data_max - data_min)),
                0, 255).astype(np.uint8)
    return data_norm

def save_video_results(output_buffer, save_path):
    video_outputs = torch.stack(output_buffer)
    for i in range(video_outputs.size(0)):
        save_name = os.path.join(save_path, 'pred_sum_{0:06d}.jpg'.format(i+9))
        imwrite(save_name, normalize_data(video_outputs[i][0].data.numpy()))

def pred_convert(y_pred):
    y_pred = np.array(y_pred)
    thr = 0.5
    y_pred_con = (y_pred > thr).astype(np.float)
    return y_pred_con


def test(data_loader, model, opt):
    print('test')

    model.eval()

    with torch.no_grad():

        data_time = AverageMeter()
        end_time = time.time()
        gt_label = []
        predict_label = []
        for i, (data, valid) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            inputs = data['rgb']

            outputs = model(inputs, data['audio'])
            outputs = outputs.cpu().data.numpy()
            gt = data['gt'].numpy()
            for i in range(outputs.size):
                predict_label.append(outputs[i][0])
                gt_label.append(gt[i])
        predict_label = pred_convert(predict_label)

        gt_label = np.array(gt_label)
        eval_res = eval_metrics(predict_label, gt_label)


def eval_metrics(y_pred, y_true):
    overlap = np.sum(y_pred * y_true)
    precision = overlap / (np.sum(y_pred) + 1e-8)
    recall = overlap / (np.sum(y_true) + 1e-8)

    if precision == 0 and recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    print('precision is {}, recall is {}, fscore is {}'.format(precision, recall, fscore))
    return [precision, recall, fscore]


