import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def check_folder(path: str):
    """Method to create directory if it doesn't exist
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_opts(opts):
    path = os.path.join(opts.out, 'opts.json')
    with open(path, 'w') as f:
        json.dump(opts.__dict__, f, indent=2)
        print("Dumped command line arguments to {}".format(path))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

# Evaluation Metrics
# Source: https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py


class PSNR(object):
    def __init__(self, des="Peak Signal to Noise Ratio"):
        self.des = des

    def __repr__(self):
        return "PSNR"

    def __call__(self, y_pred, y_true, dim=1, threshold=None):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return PSNR, larger the better
        """
        if threshold:
            y_pred = _binarize(y_pred, threshold)
        mse = torch.mean((y_pred - y_true) ** 2)
        return 10 * torch.log10(1 / mse)


class SSIM(object):
    '''
    modified from https://github.com/jorge-pessoa/pytorch-msssim
    '''

    def __init__(self, des="structural similarity index"):
        self.des = des

    def __repr__(self):
        return "SSIM"

    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - w_size // 2)**2 / float(2 * sigma**2)) for x in range(w_size)])
        return gauss / gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(
            self,
            y_pred,
            y_true,
            w_size=11,
            size_average=True,
            full=False):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            w_size : int, default 11
            size_average : boolean, default True
            full : boolean, default False
        return ssim, larger the better
        """
        # Value range can be different from 255. Other common ranges are 1
        # (sigmoid) and 2 (tanh).
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            y_pred * y_pred,
            window,
            padding=padd,
            groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(
            y_true * y_true,
            window,
            padding=padd,
            groups=channel) - mu2_sq
        sigma12 = F.conv2d(
            y_pred * y_true,
            window,
            padding=padd,
            groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret

# Source: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        # output of relu2_2
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[0:9].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).to(device)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1)).to(device)
        self.std = torch.nn.Parameter(torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1)).to(device)
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(
                input, mode='bilinear', size=(
                    224, 224), align_corners=False)
            target = self.transform(
                target, mode='bilinear', size=(
                    224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.mse_loss(x, y)
        return loss


def image_gradients(image: torch.Tensor):
    """Returns image gradients (dy, dx) for each color channel
    """
    assert image.dim() == 4, "image gradients expected the input to be 4D tensor and not {}D tensor".format(image.dim())

    # image shape
    bs, depth, height, width = image.shape

    # image gradients
    # I(x+1, y) - I(x, y)
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]

    # gradient shape as to be same as the original input, so concatenate the
    # gradient tensor with zeros
    dy = torch.cat([dy, torch.zeros((bs, depth, 1, width),
                                    device=device, dtype=image.dtype)], dim=2)
    dx = torch.cat([dx, torch.zeros((bs, depth, height, 1),
                                    device=device, dtype=image.dtype)], dim=3)

    return dy, dx


def gradient_criterion(pred, gt):
    # gradients
    gt_dy, gt_dx = image_gradients(gt)
    pred_dy, pred_dx = image_gradients(pred)

    loss = torch.mean(
        torch.abs(
            pred_dy -
            gt_dy) +
        torch.abs(
            pred_dx -
            gt_dx),
        dim=1)
    return loss
