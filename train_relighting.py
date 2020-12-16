import argparse
import os
from time import time

import numpy as np
import torch
import torchvision
from multi_illum import MultiIllum
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader

import wandb
from nets.unet import RelightModel
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

    source_probe_l1_metric = AverageMeter()
    source_probe_mse_metric = AverageMeter()
    source_probe_psnr_metric = AverageMeter()
    source_probe_ssim_metric = AverageMeter()

    relit_l1_metric = AverageMeter()
    relit_mse_metric = AverageMeter()
    relit_psnr_metric = AverageMeter()
    relit_ssim_metric = AverageMeter()

    ssim = SSIM()
    psnr = PSNR()

    with torch.no_grad():
        for didx, data in enumerate(loader):
            source_img, source_probe, target_img, target_probe = data
            source_img = source_img.to(device)
            target_img = target_img.to(device)
            source_probe = source_probe.to(device)
            target_probe = target_probe.to(device)

            source_relit, source_prob_pred, _ = net(source_img, target_probe)
            source_prob_pred = source_prob_pred.reshape(
                source_prob_pred.shape[0], 3, opts.chromesz, opts.chromesz)

            if opts.shift_range:
                source_img = torch.clamp(source_img + 0.5, 0, 1)
                source_probe = torch.clamp(source_probe + 0.5, 0, 1)
                source_prob_pred = torch.clamp(source_prob_pred + 0.5, 0, 1)
                target_img = torch.clamp(target_img + 0.5, 0, 1)
                target_probe = torch.clamp(target_probe + 0.5, 0, 1)
                source_relit = torch.clamp(source_relit + 0.5, 0, 1)

            source_probe_l1_metric.update(
                F.l1_loss(
                    source_prob_pred,
                    source_probe).item())
            source_probe_mse_metric.update(F.mse_loss(
                source_prob_pred, source_probe).item())
            source_probe_psnr_metric.update(
                psnr(source_prob_pred, source_probe).item())
            source_probe_ssim_metric.update(
                ssim(source_prob_pred, source_probe).item())

            relit_l1_metric.update(F.l1_loss(source_relit, target_img).item())
            relit_mse_metric.update(
                F.mse_loss(
                    source_relit,
                    target_img).item())
            relit_psnr_metric.update(psnr(source_relit, target_img).item())
            relit_ssim_metric.update(ssim(source_relit, target_img).item())

    return (source_probe_l1_metric.avg, source_probe_mse_metric.avg, source_probe_psnr_metric.avg,
            source_probe_ssim_metric.avg), (relit_l1_metric.avg, relit_mse_metric.avg, relit_psnr_metric.avg, relit_ssim_metric.avg)


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
    net = RelightModel(chromesz=opts.chromesz)
    wandb.watch(net)

    net.to(device)

    # dataloader
    trainset = MultiIllum(
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
        swap_channels=opts.swap_channels)
    trainloader = DataLoader(
        trainset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=5)

    valset = MultiIllum(
        datapath=os.path.join(
            opts.path,
            'val.txt'),
        cropx=opts.cropsz,
        cropy=opts.cropsz,
        probe_size=opts.chromesz,
        shift_range=opts.shift_range,
        is_train=False,
        crop_images=opts.crop_images,
        mask_probes=opts.mask_probes)
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
    best_probe_rmse, best_probe_l1, best_probe_psnr, best_probe_ssim = 1e+10, 1e+10, 1e-10, 1e-10
    best_img_rmse, best_img_l1, best_img_psnr, best_img_ssim = 1e+10, 1e+10, 1e-10, 1e-10

    # ssim loss
    ssim = SSIM()

    # training
    for epoch in range(opts.epochs):
        start_time = time()
        net.train()
        for didx, data in enumerate(trainloader):

            torch.autograd.set_detect_anomaly(True)

            source_img, source_probe, target_img, target_probe = data
            source_img = source_img.to(device)
            target_img = target_img.to(device)
            source_probe = source_probe.to(device)
            target_probe = target_probe.to(device)

            optimizer.zero_grad()

            source_relit, source_prob_pred, non_light_features = net(
                source_img, target_probe)
            source_prob_pred = source_prob_pred.reshape(
                source_prob_pred.shape[0], 3, opts.chromesz, opts.chromesz)

            # source probe loss
            probe_total_loss, probe_loss, probe_grads, probe_ssim = compute_loss(
                criterion, gradient_criterion, ssim, source_prob_pred, source_probe, opts)

            # relit image loss
            img_total_loss, img_loss, img_grads, img_ssim = compute_loss(
                criterion, gradient_criterion, ssim, source_relit, target_img, opts)

            # if non_light_loss is True
            target_probe_loss, target_loss, target_grads, target_ssim = torch.Tensor(
                [0]).to(device), torch.Tensor(
                [0]).to(device), torch.Tensor(
                [0]).to(device), torch.Tensor(
                [0]).to(device)
            non_light_feat_loss = torch.Tensor([0]).to(device)

            if opts.non_light_loss:
                _, target_prob_pred, target_non_light_features = net(
                    source_relit, None)
                target_prob_pred = target_prob_pred.reshape(
                    target_prob_pred.shape[0], 3, opts.chromesz, opts.chromesz)
                target_probe_loss, target_loss, target_grads, target_ssim = compute_loss(
                    criterion, gradient_criterion, ssim, target_prob_pred, target_probe, opts)
                non_light_feat_loss = criterion(
                    target_non_light_features, non_light_features)

            total_loss = probe_total_loss + img_total_loss + \
                target_probe_loss + non_light_feat_loss
            total_loss.backward()
            optimizer.step()

            if (didx + 1) % opts.print_frequency == 0:
                end_time = time()
                time_taken = (end_time - start_time)
                print("Epoch: [{}]/[{}], Iteration: [{}]/[{}], Total Loss: {:.4f}, Probe loss: {:.4f}, Image loss: {:.4f}, Target probe loss: {:.4f}, Non-light loss: {:.4f} Time: {:.2f}s".format(
                    epoch + 1, opts.epochs, didx + 1, len(trainloader), total_loss.item(), probe_total_loss.item(), img_total_loss.item(), target_probe_loss.item(), non_light_feat_loss.item(), time_taken))
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
                writer.add_scalar(
                    'train/probe_loss',
                    probe_total_loss.item(),
                    counter)
                wandb.log({"train/probe_loss": probe_total_loss.item()})
                writer.add_scalar(
                    'train/img_loss',
                    img_total_loss.item(),
                    counter)
                wandb.log({"train/image_loss": img_total_loss.item()})

            if (didx + 1) % opts.save_images == 0:

                # if shift range is defined, then shift all the data to be in
                # [0, 1] and clamp the values
                if opts.shift_range:
                    source_img = torch.clamp(source_img + 0.5, 0, 1)
                    source_probe = torch.clamp(source_probe + 0.5, 0, 1)
                    source_prob_pred = torch.clamp(
                        source_prob_pred + 0.5, 0, 1)
                    target_img = torch.clamp(target_img + 0.5, 0, 1)
                    target_probe = torch.clamp(target_probe + 0.5, 0, 1)
                    source_relit = torch.clamp(source_relit + 0.5, 0, 1)

                    if opts.non_light_loss:
                        target_prob_pred = torch.clamp(
                            target_prob_pred + 0.5, 0, 1)

                source_imgs = torchvision.utils.make_grid(source_img)
                source_probes = torchvision.utils.make_grid(source_probe)
                source_prob_preds = torchvision.utils.make_grid(
                    source_prob_pred)
                target_imgs = torchvision.utils.make_grid(target_img)
                target_probes = torchvision.utils.make_grid(target_probe)
                source_relits = torchvision.utils.make_grid(source_relit)

                writer.add_image('train/Source Images', source_imgs, counter)
                wandb.log(
                    {"train/source images": [wandb.Image(i) for i in source_img]})
                writer.add_image('train/Source Probes', source_probes, counter)
                wandb.log(
                    {"train/source probes": [wandb.Image(i) for i in source_probe]})
                writer.add_image(
                    'train/Source Output Probes',
                    source_prob_preds,
                    counter)
                wandb.log(
                    {"train/source output probes": [wandb.Image(i) for i in source_prob_pred]})

                writer.add_image('train/Target Images', target_imgs, counter)
                wandb.log(
                    {"train/target images": [wandb.Image(i) for i in target_img]})
                writer.add_image('train/Target Probes', target_probes, counter)
                wandb.log(
                    {"train/target probes": [wandb.Image(i) for i in target_probe]})
                writer.add_image('train/Source relit', source_relits, counter)
                wandb.log(
                    {"train/source relit": [wandb.Image(i) for i in source_relit]})

                if opts.non_light_loss:
                    target_prob_preds = torchvision.utils.make_grid(
                        target_prob_pred)
                    writer.add_image(
                        'train/Target Output Probes',
                        target_prob_preds,
                        counter)

            counter += 1
            start_time = time()

        # lr scheduler
        lr_scheduler.step()

        if (epoch + 1) % opts.validate_every == 0:

            eval_start = time()
            (probe_l1_error, probe_rmse_error, probe_psnr_value, probe_ssim_value), (img_l1_error,
                                                                                     img_rmse_error, img_psnr_value, img_ssim_value) = evaluate_model(net, valloader, device, opts)
            writer.add_scalar('val/probe_mae', probe_l1_error, epoch + 1)
            wandb.log({"val/probe_mae": probe_l1_error})
            writer.add_scalar('val/probe_rmse', probe_rmse_error, epoch + 1)
            wandb.log({"val/probe_rmse": probe_rmse_error})
            writer.add_scalar('val/probe_psnr', probe_psnr_value, epoch + 1)
            wandb.log({"val/probe_psnr": probe_psnr_value})
            writer.add_scalar('val/probe_ssim', probe_ssim_value, epoch + 1)
            wandb.log({"val/probe_ssim": probe_ssim_value})
            writer.add_scalar('val/img_mae', img_l1_error, epoch + 1)
            wandb.log({"val/img_mae": img_l1_error})
            writer.add_scalar('val/img_rmse', img_rmse_error, epoch + 1)
            wandb.log({"val/img_rmse": img_rmse_error})
            writer.add_scalar('val/img_psnr', img_psnr_value, epoch + 1)
            wandb.log({"val/img_psnr": img_psnr_value})
            writer.add_scalar('val/img_ssim', img_ssim_value, epoch + 1)
            wandb.log({"val/img_ssim": img_ssim_value})

            print(
                "Val Epoch: [{}]/[{}], Probe MAE: {:.4f}, RMSE: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}, Time: {:.2f}s".format(
                    epoch +
                    1,
                    opts.epochs,
                    probe_l1_error,
                    probe_rmse_error,
                    probe_psnr_value,
                    probe_ssim_value,
                    time() -
                    eval_start))
            print("Val Epoch: [{}]/[{}], Image MAE: {:.4f}, RMSE: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}, Time: {:.2f}s".format(
                epoch + 1, opts.epochs, img_l1_error, img_rmse_error, img_psnr_value, img_ssim_value, time() - eval_start))

            # save best model
            if probe_rmse_error < best_probe_rmse or probe_l1_error < best_probe_l1 or probe_psnr_value > best_probe_psnr or probe_ssim_value > best_probe_ssim or img_rmse_error < best_img_rmse or img_l1_error < best_img_l1 or img_psnr_value > best_img_psnr or img_ssim_value > best_img_ssim:
                model_path = os.path.join(opts.out, "models")
                check_folder(model_path)
                model_name = "epoch_{}_pmae_{:.2f}_prmse_{:.2f}_ppsnr_{:.2f}_pssim_{:.2f}_imae_{:.2f}_irmse_{:.2f}_ipsnr_{:.2f}_issim_{:.2f}.pth".format(
                    epoch + 1, probe_l1_error, probe_rmse_error, probe_psnr_value, probe_ssim_value, img_l1_error, img_rmse_error, img_psnr_value, img_ssim_value)
                model_snapshot_name = os.path.join(model_path, model_name)
                torch.save(net.state_dict(), model_snapshot_name)

            # capturing the best values after model valiation
            if probe_l1_error < best_probe_l1:
                best_probe_l1 = probe_l1_error
            if probe_rmse_error < best_probe_rmse:
                best_probe_rmse = probe_rmse_error
            if probe_psnr_value > best_probe_psnr:
                best_probe_psnr = probe_psnr_value
            if probe_ssim_value > best_probe_ssim:
                best_probe_ssim = probe_ssim_value
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
    parser.add_argument("--non_light_loss", action='store_true')
    parser.add_argument("--swap_channels", action='store_true')
    parser.add_argument("--model", type=str, default="original")

    parser.add_argument("--print_frequency", type=int, default=30)
    parser.add_argument("--save_images", type=int, default=300)
    parser.add_argument("--validate_every", type=int, default=1)

    opts = parser.parse_args()

    main(opts)
