import argparse
import os
from time import time

import numpy as np
import torch
import torchvision
from multi_illum import MultiIllumRelightingBaseline
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader

import wandb
from nets.unet import ResNetUNet
from utils import (PSNR, SSIM, AverageMeter, check_folder, gradient_criterion,
                   save_opts)


def compute_loss(
        criterion,
        gradient_criterion,
        ssim,
        prediction,
        ground_truth,
        opts):
    """Return total loss between prediction and ground truth image

    Args:
        criterion: L1 loss
        gradient_criterion: gradient loss (edges) between prediction and groundtruth
        ssim: SSIM loss
        prediction: predicted image
        ground_truth: groundtruth image
        opts: options (argparse)

    Returns:
        total_loss
    """
    loss = criterion(prediction, ground_truth)
    loss_edges = torch.mean(gradient_criterion(prediction, ground_truth))
    ssim_loss = torch.clamp((1 - ssim(prediction, ground_truth)), 0, 1)
    total_loss = (opts.theta * loss) + (1.0 * loss_edges) + (1.0 * ssim_loss)
    return total_loss, loss, loss_edges, ssim_loss


def evaluate_model(net, loader, device, opts):
    net.eval()

    relit_l1_metric = AverageMeter()
    relit_mse_metric = AverageMeter()
    relit_psnr_metric = AverageMeter()
    relit_ssim_metric = AverageMeter()

    ssim = SSIM()
    psnr = PSNR()

    with torch.no_grad():
        for didx, data in enumerate(loader):
            source_img, target_img = data
            source_img = source_img.to(device)
            target_img = target_img.to(device)

            source_relit = net(source_img)

            if opts.shift_range:
                source_img = torch.clamp(source_img + 0.5, 0, 1)
                target_img = torch.clamp(target_img + 0.5, 0, 1)
                source_relit = torch.clamp(source_relit + 0.5, 0, 1)

            relit_l1_metric.update(F.l1_loss(source_relit, target_img).item())
            relit_mse_metric.update(
                F.mse_loss(
                    source_relit,
                    target_img).item())
            relit_psnr_metric.update(psnr(source_relit, target_img).item())
            relit_ssim_metric.update(ssim(source_relit, target_img).item())

    return relit_l1_metric.avg, relit_mse_metric.avg, relit_psnr_metric.avg, relit_ssim_metric.avg


def main(opts):

    # wandb init
    wandb.init(config=vars(opts))

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
    net = ResNetUNet()
    wandb.watch(net)

    net.to(device)

    # dataloader
    trainset = MultiIllumRelightingBaseline(
        datapath=os.path.join(
            opts.path,
            'train.txt'),
        cropx=opts.cropsz,
        cropy=opts.cropsz,
        probe_size=opts.chromesz,
        shift_range=opts.shift_range,
        is_train=True,
        crop_images=opts.crop_images,
        mask_probes=opts.mask_probes,
        swap_channels=opts.swap_channels,
        random_relight=opts.random_relight)
    trainloader = DataLoader(
        trainset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=5)

    # TODO: fix is_train=False for validation and make changes to accomodate
    # full image resolution (H: 1000, W: 1500)
    valset = MultiIllumRelightingBaseline(
        datapath=os.path.join(
            opts.path,
            'val.txt'),
        cropx=opts.cropsz,
        cropy=opts.cropsz,
        probe_size=opts.chromesz,
        shift_range=opts.shift_range,
        is_train=False,
        crop_images=opts.crop_images,
        mask_probes=opts.mask_probes,
        random_relight=opts.random_relight)
    valloader = DataLoader(
        valset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=5)

    # optimizer and criterion
    optimizer = torch.optim.Adam(net.parameters(), lr=opts.learning_rate)
    if opts.loss_type == 'mse':
        criterion = torch.nn.MSELoss()
    elif opts.loss_type == 'l1':
        criterion = torch.nn.L1Loss()

    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15)

    # tensorboard counter
    counter = 1

    # best values
    best_img_rmse, best_img_l1, best_img_psnr, best_img_ssim = 1e+10, 1e+10, 1e-10, 1e-10

    # ssim loss
    ssim = SSIM()

    # training
    for epoch in range(opts.epochs):
        start_time = time()
        net.train()
        for didx, data in enumerate(trainloader):

            source_img, target_img = data
            source_img = source_img.to(device)
            target_img = target_img.to(device)

            optimizer.zero_grad()
            source_relit = net(source_img)

            # relit image loss
            img_total_loss, img_loss, img_grads, img_ssim = compute_loss(
                criterion, gradient_criterion, ssim, source_relit, target_img, opts)

            total_loss = img_total_loss
            total_loss.backward()
            optimizer.step()

            if (didx + 1) % opts.print_frequency == 0:
                end_time = time()
                time_taken = (end_time - start_time)
                print("Epoch: [{}]/[{}], Iteration: [{}]/[{}], Total Loss: {:.4f}, Time: {:.2f}s".format(
                    epoch + 1, opts.epochs, didx + 1, len(trainloader), total_loss.item(), time_taken))
                writer.add_scalar(
                    'train/Learning Rate',
                    optimizer.param_groups[0]['lr'],
                    counter)
                wandb.log({"train/lr": optimizer.param_groups[0]['lr']})
                writer.add_scalar(
                    'train/total_loss',
                    total_loss.item(),
                    counter)
                wandb.log({"train/total_loss": total_loss.item()})

            if (didx + 1) % opts.save_images == 0:

                # if shift range is defined, then shift all the data to be in
                # [0, 1] and clamp the values
                if opts.shift_range:
                    source_img = torch.clamp(source_img + 0.5, 0, 1)
                    target_img = torch.clamp(target_img + 0.5, 0, 1)
                    source_relit = torch.clamp(source_relit + 0.5, 0, 1)

                source_imgs = torchvision.utils.make_grid(source_img)
                target_imgs = torchvision.utils.make_grid(target_img)
                source_relits = torchvision.utils.make_grid(source_relit)

                writer.add_image('train/Source Images', source_imgs, counter)
                wandb.log(
                    {"train/source images": [wandb.Image(i) for i in source_img]})
                writer.add_image('train/Target Images', target_imgs, counter)
                wandb.log(
                    {"train/target images": [wandb.Image(i) for i in target_img]})
                writer.add_image('train/Source relit', source_relits, counter)
                wandb.log(
                    {"train/source relit": [wandb.Image(i) for i in source_relit]})

            counter += 1
            start_time = time()

        # lr scheduler
        lr_scheduler.step()

        if (epoch + 1) % opts.validate_every == 0:

            eval_start = time()
            img_l1_error, img_rmse_error, img_psnr_value, img_ssim_value = evaluate_model(
                net, valloader, device, opts)
            writer.add_scalar('val/img_mae', img_l1_error, epoch + 1)
            wandb.log({"val/img_mae": img_l1_error})
            writer.add_scalar('val/img_rmse', img_rmse_error, epoch + 1)
            wandb.log({"val/img_rmse": img_rmse_error})
            writer.add_scalar('val/img_psnr', img_psnr_value, epoch + 1)
            wandb.log({"val/img_psnr": img_psnr_value})
            writer.add_scalar('val/img_ssim', img_ssim_value, epoch + 1)
            wandb.log({"val/img_ssim": img_ssim_value})
            print("Val Epoch: [{}]/[{}], Image MAE: {:.4f}, RMSE: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}, Time: {:.2f}s".format(
                epoch + 1, opts.epochs, img_l1_error, img_rmse_error, img_psnr_value, img_ssim_value, time() - eval_start))

            # save best model
            if img_rmse_error < best_img_rmse or img_l1_error < best_img_l1 or img_psnr_value > best_img_psnr or img_ssim_value > best_img_ssim:
                model_path = os.path.join(opts.out, "models")
                check_folder(model_path)
                model_name = "epoch_{}_imae_{:.2f}_irmse_{:.2f}_ipsnr_{:.2f}_issim_{:.2f}.pth".format(
                    epoch + 1, img_l1_error, img_rmse_error, img_psnr_value, img_ssim_value)
                model_snapshot_name = os.path.join(model_path, model_name)
                torch.save(net.state_dict(), model_snapshot_name)

            # capturing the best values after model valiation
            if img_l1_error < best_img_l1:
                best_img_l1 = img_l1_error
            if img_rmse_error < best_img_rmse:
                best_img_rmse = img_rmse_error
            if img_psnr_value > best_img_psnr:
                best_img_psnr = img_psnr_value
            if img_ssim_value > best_img_ssim:
                best_img_ssim = img_ssim_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='./data')
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
    parser.add_argument("--random_relight", action='store_true')

    parser.add_argument("--print_frequency", type=int, default=30)
    parser.add_argument("--save_images", type=int, default=300)
    parser.add_argument("--validate_every", type=int, default=1)

    opts = parser.parse_args()

    main(opts)
