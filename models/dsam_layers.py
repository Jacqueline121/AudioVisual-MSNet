import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def center_crop(x, height, width):
    crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
    crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)

    # fixed indexing for PyTorch 0.4
    return F.pad(x, [int(crop_w.ceil()[0]), int(crop_w.floor()[0]), int(crop_h.ceil()[0]), int(crop_h.floor()[0])])

class sa(nn.Module):
    def __init__(self, prev_layer, prev_nfilters, prev_nsamples):
        super(dsam_score_dsn, self).__init__()
        i = prev_layer
        self.avgpool = nn.AvgPool3d((prev_nsamples, 1, 1), stride=1)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x, crop_h, crop_w):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, num_samples, width, height = x.size()


        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

class dsam_score_dsn(nn.Module):

    def __init__(self, prev_layer, prev_nfilters, prev_nsamples):

        super(dsam_score_dsn, self).__init__()
        i = prev_layer
        self.avgpool = nn.AvgPool3d((prev_nsamples, 1, 1), stride=1)
        # Make the layers of the preparation step
        self.side_prep = nn.Conv2d(prev_nfilters, 16, kernel_size=3, padding=1)
        # Make the layers of the score_dsn step
        self.score_dsn = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        self.upscale_ = nn.ConvTranspose2d(1, 1, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False)
        self.upscale = nn.ConvTranspose2d(16, 16, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False)

    def forward(self, x, crop_h, crop_w):

        self.crop_h = crop_h
        self.crop_w = crop_w
        x = self.avgpool(x).squeeze(2)  # [64, 64, 56, 56]
        side_temp = self.side_prep(x)  # [64, 16, 56, 56]
        side = center_crop(self.upscale(side_temp), self.crop_h, self.crop_w)
        side_out_tmp = self.score_dsn(side_temp)  # [64, 1, 56, 56]
        side_out = center_crop(self.upscale_(side_out_tmp), self.crop_h, self.crop_w)  # [64, 1, 112, 112]
        return side, side_out, side_out_tmp


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def spatial_softmax(x):
    x = torch.exp(x)
    sum_batch = torch.sum(torch.sum(x, 2, keepdim=True), 3, keepdim=True)
    x_soft = torch.div(x,sum_batch)
    return x_soft

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# this is for deconvolution without groups
def interp_surgery(lay):
        m, k, h, w = lay.weight.data.size()
        if m != k:
            print('input + output channels need to be the same')
            raise ValueError
        if h != w:
            print('filters need to be square')
            raise ValueError
        filt = upsample_filt(h)

        for i in range(m):
            lay.weight[i, i, :, :].data.copy_(torch.from_numpy(filt))

        return lay.weight.data