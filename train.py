import argparse
import os
import urllib
from time import time

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader

from multi_illum import MultiIllum
from nets import model
from nets.illum_nets import VGG16, ResNet18
from nets.unet import UnetEnvMap
from utils import *


def evaluate_model(net, loader, device, opts):
    net.eval()
    
    l1_metric = AverageMeter()
    mse_metric = AverageMeter()
    psnr_metric = AverageMeter()
    ssim_metric = AverageMeter()

    ssim = SSIM()
    psnr = PSNR()

    with torch.no_grad():
        for didx, data in enumerate(loader):
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

    return l1_metric.avg, np.sqrt(mse_metric.avg), psnr_metric.avg, ssim_metric.avg

def main(opts):

    # seed 
    torch.manual_seed(0)
    np.random.seed(0)

    # tensorboard writer
    check_folder(opts.out)
    writer = SummaryWriter(opts.out)
    
    # save opts
    save_opts(opts)

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

    # dataloader
    # def __init__(self, datapath: str, cropx: int, cropy: int, probe_size: int, shift_range: bool = False, is_train: bool = True, crop_images: bool = False, mask_probes: bool = False):
    trainset = MultiIllum(datapath=os.path.join(opts.path, 'train.txt'), cropx=opts.cropsz, cropy=opts.cropsz, probe_size=opts.chromesz, 
                                                shift_range=opts.shift_range, is_train=True, crop_images=opts.crop_images, mask_probes=opts.mask_probes, swap_channels=opts.swap_channels)
    trainloader = DataLoader(trainset, batch_size=opts.batch_size, shuffle=True, num_workers=5)

    # TODO: fix is_train=False for validation and make changes to accomodate full image resolution (H: 1000, W: 1500)
    valset = MultiIllum(datapath=os.path.join(opts.path, 'val.txt'), cropx=opts.cropsz, cropy=opts.cropsz, probe_size=opts.chromesz, 
                                                shift_range=opts.shift_range, is_train=False, crop_images=opts.crop_images, mask_probes=opts.mask_probes)
    valloader = DataLoader(valset, batch_size=opts.batch_size, shuffle=False, num_workers=5)

    # optimizer and criterion
    optimizer = torch.optim.Adam(net.parameters(), lr=opts.learning_rate)
    if opts.loss_type == 'mse':
        criterion = torch.nn.MSELoss()
    elif opts.loss_type == 'l1':
        criterion = torch.nn.L1Loss()
    
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)

    # tensorboard counter
    counter = 1

    # best values
    best_rmse, best_l1, best_psnr, best_ssim = 1e+10, 1e+10, 1e-10, 1e-10

    # ssim loss
    ssim = SSIM()
    perceptual = VGGPerceptualLoss(resize=False)

    # training
    for epoch in range(opts.epochs):
        start_time = time()
        net.train()
        for didx, data in enumerate(trainloader):
            img, probe, _, _ = data
            img = img.to(device)
            probe = probe.to(device)

            optimizer.zero_grad()

            pred = net(img)
            pred = pred.reshape(pred.shape[0], 3, opts.chromesz, opts.chromesz)

            loss = criterion(pred, probe)
            loss_edges = torch.mean(gradient_criterion(pred, probe))
            ssim_loss = torch.clamp((1 - ssim(pred, probe)), 0, 1)
            total_loss = (opts.theta * loss) + (1.0 * loss_edges) + (1.0 * ssim_loss)
            total_loss.backward()
            optimizer.step()

            if (didx + 1) % opts.print_frequency == 0:
                end_time = time()
                time_taken = (end_time - start_time)
                print("Epoch: [{}]/[{}], Iteration: [{}]/[{}], Total Loss: {:.4f}, Loss: {:.4f}, SSIM: {:.4f}, Gradient Loss: {:.4f}, Time: {:.2f}s".format(epoch + 1, opts.epochs, 
                                                                    didx + 1, len(trainloader), total_loss.item(), loss.item(), ssim_loss.item(), loss_edges.item(), time_taken))
                writer.add_scalar('train/Loss', loss.item(), counter)
                writer.add_scalar('train/Learning Rate', optimizer.param_groups[0]['lr'], counter)

                if (didx + 1) % opts.save_images == 0:
                    
                    # if shift range is defined, then shift all the data to be in [0, 1] and clamp the values
                    if opts.shift_range:
                        img = torch.clamp(img + 0.5, 0, 1)
                        probe = torch.clamp(probe + 0.5, 0, 1)
                        pred = torch.clamp(pred + 0.5, 0, 1)

                    input_imgs = torchvision.utils.make_grid(img)
                    input_probes = torchvision.utils.make_grid(probe)
                    output_probes = torchvision.utils.make_grid(pred)

                    writer.add_image('train/Input Images', input_imgs, counter)
                    writer.add_image('train/Input Probes', input_probes, counter)
                    writer.add_image('train/Output Probes', output_probes, counter)

                counter += 1
                start_time = time()
        
        # lr scheduler
        lr_scheduler.step()
        
        if (epoch + 1) % opts.validate_every == 0:
            
            eval_start = time()
            l1_error, rmse_error, psnr_value, ssim_value = evaluate_model(net, valloader, device, opts)
            writer.add_scalar('val/mae', l1_error, epoch + 1)
            writer.add_scalar('val/rmse', rmse_error, epoch + 1)
            writer.add_scalar('val/psnr', psnr_value, epoch + 1)
            writer.add_scalar('val/ssim', ssim_value, epoch + 1)

            print("Val Epoch: [{}]/[{}], MAE: {:.4f}, RMSE: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f} Time: {:.2f}s".format(epoch + 1, opts.epochs, l1_error, rmse_error, psnr_value, ssim_value, time()-eval_start))

            # save best model
            if rmse_error < best_rmse or l1_error < best_l1 or psnr_value > best_psnr or ssim_value > best_ssim:
                model_path = os.path.join(opts.out, "models")
                check_folder(model_path)
                model_snapshot_name = os.path.join(model_path, "epoch_{}_mae_{:.2f}_rmse_{:.2f}_psnr_{:.2f}_ssim_{:.2f}.pth".format(epoch + 1, l1_error, rmse_error, psnr_value, ssim_value))
                torch.save(net.state_dict(), model_snapshot_name)
            
            # capturing the best values after model valiation
            if l1_error < best_l1:
                best_l1 = l1_error
            if rmse_error < best_rmse:
                best_rmse = rmse_error
            if psnr_value > best_psnr:
                best_psnr = psnr_value
            if ssim_value > best_ssim:
                best_ssim = ssim_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('-o', '--out', type=str)

    parser.add_argument("--cropx", type=int, default=512)
    parser.add_argument("--cropy", type=int, default=256)

    parser.add_argument("--cropsz", type=int, default=512)
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--chromesz", type=int, default=64)
    parser.add_argument("--fmaps1", type=int, default=6)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--loss_type", choices=["mse", "l1"])
    parser.add_argument("--shift_range", action='store_true')
    parser.add_argument("--crop_images", action='store_true')
    parser.add_argument("--mask_probes", action='store_true')
    parser.add_argument("--swap_channels", action='store_true')
    parser.add_argument("--model", type=str, default="original")

    parser.add_argument("--print_frequency", type=int, default=30)
    parser.add_argument("--save_images", type=int, default=300)
    parser.add_argument("--validate_every", type=int, default=1)

    opts = parser.parse_args()

    main(opts)
