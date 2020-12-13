import argparse
import os

import model
import numpy as np
import torch
from skimage.io import imsave
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from multi_illum import MultiIllum
from nets.illum_nets import VGG16, ResNet18
from nets.unet import UnetEnvMap
from utils import PSNR, SSIM, AverageMeter, check_folder


def main(opts):

    # seed
    torch.manual_seed(0)
    np.random.seed(0)

    # tensorboard writer
    check_folder(opts.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    if opts.model == 'vgg':
        net = VGG16(chromesz=opts.chromesz)
    if opts.model == 'resnet18':
        net = ResNet18(chromesz=opts.chromesz)
    elif opts.model == 'original':
        net = model.make_net_fully_convolutional(
            chromesz=opts.chromesz,
            fmaps1=opts.fmaps1,
            xysize=opts.cropsz)
    elif opts.model == 'unet':
        net = UnetEnvMap(chromesz=opts.chromesz)

    net.to(device)

    if opts.weights is not None:
        checkpoint = torch.load(opts.weights)
        net.load_state_dict(checkpoint)

    # dataloader
    testset = MultiIllum(
        datapath=os.path.join(
            opts.path,
            'test.txt'),
        cropx=opts.cropsz,
        cropy=opts.cropsz,
        probe_size=opts.chromesz,
        shift_range=opts.shift_range,
        is_train=False,
        crop_images=opts.crop_images,
        mask_probes=opts.mask_probes)
    testloader = DataLoader(
        testset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=5,
        pin_memory=True)

    net.eval()

    l1_metric = AverageMeter()
    mse_metric = AverageMeter()
    psnr_metric = AverageMeter()
    ssim_metric = AverageMeter()

    ssim = SSIM()
    psnr = PSNR()

    with torch.no_grad():
        for didx, data in tqdm(enumerate(testloader), total=len(testloader)):
            img, probe, _, _ = data
            img = img.to(device)
            probe = probe.to(device)

            pred = net(img)
            pred = pred.reshape(pred.shape[0], 3, opts.chromesz, opts.chromesz)

            if opts.shift_range:
                img = torch.clamp(img + 0.5, 0, 1)
                probe = torch.clamp(probe + 0.5, 0, 1)
                pred = torch.clamp(pred + 0.5, 0, 1)

            l1_metric.update(F.l1_loss(pred, probe).item())
            mse_metric.update(F.mse_loss(pred, probe).item())
            psnr_metric.update(psnr(pred, probe).item())
            ssim_metric.update(ssim(pred, probe).item())

            img_path = os.path.join(opts.out, 'img_{}.png'.format(str(didx).zfill(3)))
            gt_probe_path = os.path.join(opts.out, 'gt_probe_{}.png'.format(str(didx).zfill(3)))
            pred_probe_path = os.path.join(opts.out, 'pred_probe_{}.png'.format(str(didx).zfill(3)))

            img = (img.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
            probe = (probe.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
            pred = (pred.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)

            imsave(img_path, img)
            imsave(gt_probe_path, probe)
            imsave(pred_probe_path, pred)

    print("Evaluation metrics: L1: {}, RMSE: {}, PSNR: {}, SSIM: {}".format(
        l1_metric.avg, np.sqrt(mse_metric.avg), psnr_metric.avg, ssim_metric.avg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='./data')
    parser.add_argument('-o', '--out', type=str)

    parser.add_argument("--cropx", type=int, default=512)
    parser.add_argument("--cropy", type=int, default=256)

    parser.add_argument('--weights', type=str, default=None)

    parser.add_argument("--cropsz", type=int, default=512)
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--chromesz", type=int, default=64)
    parser.add_argument("--fmaps1", type=int, default=6)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shift_range", action='store_true')
    parser.add_argument("--crop_images", action='store_true')
    parser.add_argument("--mask_probes", action='store_true')
    parser.add_argument("--swap_channels", action='store_true')
    parser.add_argument("--model", type=str, default="original")

    opts = parser.parse_args()

    main(opts)
