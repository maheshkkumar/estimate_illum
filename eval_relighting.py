import argparse
import json
import os

import numpy as np
import torch
from skimage.io import imsave
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.multi_illum import TestMultiIllumRelighting
from nets.unet import RelightModel, ResNetUNet, UnetEnvMap
from utils import PSNR, SSIM, AverageMeter, check_folder

ssim = SSIM()
psnr = PSNR()


def compute_metrics(
        source_img,
        target_img,
        source_relit,
        source_probe,
        target_probe,
        source_probe_pred,
        opts):
    if opts.shift_range:
        source_img = torch.clamp(source_img + 0.5, 0, 1)
        source_probe = torch.clamp(source_probe + 0.5, 0, 1)
        source_probe_pred = torch.clamp(source_probe_pred + 0.5, 0, 1)
        target_img = torch.clamp(target_img + 0.5, 0, 1)
        target_probe = torch.clamp(target_probe + 0.5, 0, 1)
        source_relit = torch.clamp(source_relit + 0.5, 0, 1)

    l1_value = F.l1_loss(source_relit, target_img).item()
    mse_value = F.mse_loss(source_relit, target_img).item()
    psnr_value = psnr(source_relit, target_img).item()
    ssim_value = ssim(source_relit, target_img).item()

    return (l1_value, mse_value, psnr_value, ssim_value), (source_img,
                                                           target_img, source_relit, source_probe, target_probe, source_probe_pred)


def compute_metrics_baseline(
        source_img,
        target_img,
        source_relit,
        opts):
    if opts.shift_range:
        source_img = torch.clamp(source_img + 0.5, 0, 1)
        target_img = torch.clamp(target_img + 0.5, 0, 1)
        source_relit = torch.clamp(source_relit + 0.5, 0, 1)

    l1_value = F.l1_loss(source_relit, target_img).item()
    mse_value = F.mse_loss(source_relit, target_img).item()
    psnr_value = psnr(source_relit, target_img).item()
    ssim_value = ssim(source_relit, target_img).item()

    return (l1_value, mse_value, psnr_value,
            ssim_value), (source_img, target_img, source_relit)


def main(opts):

    # seed
    torch.manual_seed(0)
    np.random.seed(0)

    # tensorboard writer
    check_folder(opts.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    if opts.models == 'unet':
        net = ResNetUNet()
    else:
        net = RelightModel(chromesz=opts.chromesz)
    net.to(device)

    # load pre-trained weights for evaluation
    if opts.weights is not None:
        checkpoint = torch.load(opts.weights)
        net.load_state_dict(checkpoint)

    # dataloader
    source_testset = TestMultiIllumRelighting(
        datapath=opts.source_scene,
        cropx=opts.cropsz,
        cropy=opts.cropsz,
        probe_size=opts.chromesz,
        shift_range=opts.shift_range,
        is_train=False,
        crop_images=opts.crop_images,
        mask_probes=opts.mask_probes)
    source_testloader = DataLoader(
        source_testset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=0)

    # dataloader
    target_testset = TestMultiIllumRelighting(
        datapath=opts.target_scene,
        cropx=opts.cropsz,
        cropy=opts.cropsz,
        probe_size=opts.chromesz,
        shift_range=opts.shift_range,
        is_train=False,
        crop_images=opts.crop_images,
        mask_probes=opts.mask_probes)
    target_testloader = DataLoader(
        target_testset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=0)

    net.eval()

    relit_l1_metric = AverageMeter()
    relit_mse_metric = AverageMeter()
    relit_psnr_metric = AverageMeter()
    relit_ssim_metric = AverageMeter()

    relighting_images = {}
    relighting_metrics = {}
    with torch.no_grad():
        for didx, (source_data, target_data) in enumerate(
                zip(source_testloader, target_testloader)):
            source_imgs, source_probes, source_scene_name = source_data
            target_imgs, target_probes, target_scene_name = target_data

            source_scene_name = source_scene_name[0]
            target_scene_name = target_scene_name[0]
            if opts.reference_probe:
                scene_name = '{}_{}'.format(
                    source_scene_name, target_scene_name)
            else:
                scene_name = '{}_{}_no_probe'.format(
                    source_scene_name, target_scene_name)

            source_imgs = source_imgs.to(device)
            source_probes = source_probes.to(device)
            target_imgs = target_imgs.to(device)
            target_probes = target_probes.to(device)

            iB, iN, iC, iH, iW = source_imgs.shape
            pB, pN, pC, pH, pW = source_probes.shape
            source_imgs = source_imgs.reshape((iB * iN, iC, iH, iW))
            source_probes = source_probes.reshape((pB * pN, pC, pH, pW))
            target_imgs = target_imgs.reshape((iB * iN, iC, iH, iW))
            target_probes = target_probes.reshape((pB * pN, pC, pH, pW))

            if scene_name not in relighting_images.keys():
                relighting_images[scene_name] = []
            if scene_name not in relighting_metrics.keys():
                relighting_metrics[scene_name] = []

            # using target probe to relight the source image
            if opts.reference_probe:
                if not opts.baseline:
                    for source_idx in tqdm(range(iB * iN)):
                        for target_idx in range(iB * iN):
                            source_img, source_probe = source_imgs[source_idx], source_probes[source_idx]
                            target_img, target_probe = target_imgs[target_idx], target_probes[target_idx]

                            source_img = source_img.unsqueeze(0)
                            source_probe = source_probe.unsqueeze(0)
                            target_img = target_img.unsqueeze(0)
                            target_probe = target_probe.unsqueeze(0)

                            source_relit, source_probe_pred, _ = net(
                                source_img, target_probe)
                            source_probe_pred = source_probe_pred.reshape(
                                source_probe_pred.shape[0], 3, opts.chromesz, opts.chromesz)

                            (l1_value, mse_value, psnr_value, ssim_value), (source_img, target_img, source_relit, source_probe, target_probe,
                                                                            source_probe_pred) = compute_metrics(source_img, target_img, source_relit, source_probe, target_probe, source_probe_pred, opts)

                            relit_l1_metric.update(l1_value)
                            relit_mse_metric.update(mse_value)
                            relit_psnr_metric.update(psnr_value)
                            relit_ssim_metric.update(ssim_value)

                            relighting_images[scene_name].append(
                                [
                                    (source_img.squeeze(0).cpu().numpy() *
                                     255.).astype(
                                        np.uint8),
                                    (target_img.squeeze(0).cpu().numpy() *
                                     255.).astype(
                                        np.uint8),
                                    (source_relit.squeeze(0).cpu().numpy() *
                                     255.).astype(
                                        np.uint8),
                                    (source_probe.squeeze(0).cpu().numpy() *
                                     255.).astype(
                                        np.uint8),
                                    (target_probe.squeeze(0).cpu().numpy() *
                                     255.).astype(
                                        np.uint8),
                                    (source_probe_pred.squeeze(0).cpu().numpy() *
                                     255.).astype(
                                        np.uint8),
                                    '{}_{}'.format(
                                        str(source_idx).zfill(2),
                                        str(target_idx).zfill(2))])
                # using target image to estimate the illumination and then use it
                # to relight the source image
                # baseline models
                else:
                    for source_idx in range(iB * iN):
                        for target_idx in range(iB * iN):
                            source_img, source_probe = source_imgs[source_idx], source_probes[source_idx]
                            target_img, target_probe = target_imgs[target_idx], target_probes[target_idx]

                            source_img = source_img.unsqueeze(0)
                            target_img = target_img.unsqueeze(0)

                            source_relit = net(source_img)

                            (l1_value, mse_value, psnr_value, ssim_value), (source_img, target_img,
                                                                            source_relit) = compute_metrics_baseline(source_img, target_img, source_relit, opts)

                            relit_l1_metric.update(l1_value)
                            relit_mse_metric.update(mse_value)
                            relit_psnr_metric.update(psnr_value)
                            relit_ssim_metric.update(ssim_value)

                            # relighting_images[scene_name].append(
                            #     [
                            #         (source_img.squeeze(0).cpu().numpy() * 255.).astype(np.uint8),
                            #         (target_img.squeeze(0).cpu().numpy() * 255.).astype(np.uint8),
                            #         (source_relit.squeeze(0).cpu().numpy() * 255.).astype(np.uint8),
                            #         '{}_{}'.format(str(source_idx).zfill(2), str(target_idx).zfill(2))
                            #     ])
            else:
                print("Using target reference images to relight the source image")
                if opts.illum_model == 'unet':
                    illum_net = UnetEnvMap(chromesz=opts.chromesz)

                checkpoint = torch.load(opts.illum_weights)
                illum_net.load_state_dict(checkpoint)
                illum_net.to(device)

                illum_net.eval()

                for source_idx in tqdm(range(iB * iN)):
                    for target_idx in range(iB * iN):
                        source_img, source_probe = source_imgs[source_idx], source_probes[source_idx]
                        target_img, _ = target_imgs[target_idx], target_probes[target_idx]

                        source_img = source_img.unsqueeze(0)
                        source_probe = source_probe.unsqueeze(0)
                        target_img = target_img.unsqueeze(0)
                        # target_probe = target_probe.unsqueeze(0)

                        target_probe = illum_net(target_img)
                        target_probe = target_probe.reshape(
                            target_probe.shape[0], 3, opts.chromesz, opts.chromesz)

                        source_relit, source_probe_pred, _ = net(
                            source_img, target_probe)
                        source_probe_pred = source_probe_pred.reshape(
                            source_probe_pred.shape[0], 3, opts.chromesz, opts.chromesz)

                        (l1_value, mse_value, psnr_value, ssim_value), (source_img, target_img, source_relit, source_probe, target_probe,
                                                                        source_probe_pred) = compute_metrics(source_img, target_img, source_relit, source_probe, target_probe, source_probe_pred, opts)

                        relit_l1_metric.update(l1_value)
                        relit_mse_metric.update(mse_value)
                        relit_psnr_metric.update(psnr_value)
                        relit_ssim_metric.update(ssim_value)

                        relighting_images[scene_name].append(
                            [
                                (source_img.squeeze(0).cpu().numpy() *
                                 255.).astype(
                                    np.uint8),
                                (target_img.squeeze(0).cpu().numpy() *
                                 255.).astype(
                                    np.uint8),
                                (source_relit.squeeze(0).cpu().numpy() *
                                 255.).astype(
                                    np.uint8),
                                (source_probe.squeeze(0).cpu().numpy() *
                                 255.).astype(
                                    np.uint8),
                                (target_probe.squeeze(0).cpu().numpy() *
                                 255.).astype(
                                    np.uint8),
                                (source_probe_pred.squeeze(0).cpu().numpy() *
                                 255.).astype(
                                    np.uint8),
                                '{}_{}'.format(
                                    str(source_idx).zfill(2),
                                    str(target_idx).zfill(2))])
        print(
            "Evaluation metrics: L1: {}, RMSE: {}, PSNR: {}, SSIM: {}".format(
                relit_l1_metric.avg,
                np.sqrt(relit_mse_metric.avg),
                relit_psnr_metric.avg,
                relit_ssim_metric.avg))

        # saving images
        for scene, imgs in tqdm(relighting_images.items()):
            for idx, img in enumerate(imgs):
                img_name = img[-1]
                img = [i.transpose(1, 2, 0) for i in img[:-1]]
                si, ti, sr, sp, tp, sprp = img

                folder_name = os.path.join(opts.out, scene_name)
                check_folder(folder_name)

                imsave(
                    os.path.join(
                        folder_name,
                        'si_{}.png'.format(img_name)),
                    si)
                imsave(
                    os.path.join(
                        folder_name,
                        'ti_{}.png'.format(img_name)),
                    ti)
                imsave(
                    os.path.join(
                        folder_name,
                        'sp_{}.png'.format(img_name)),
                    sp)
                imsave(
                    os.path.join(
                        folder_name,
                        'tp_{}.png'.format(img_name)),
                    tp)
                imsave(
                    os.path.join(
                        folder_name,
                        'sr_{}.png'.format(img_name)),
                    sr)
                imsave(
                    os.path.join(
                        folder_name,
                        'sprp_{}.png'.format(img_name)),
                    sprp)

        # saving metrics
        with open(os.path.join(opts.out, 'metrics.json'), 'w') as fp:
            json.dump(relighting_metrics, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='./data')
    parser.add_argument('-o', '--out', type=str)

    # source and target scenes
    parser.add_argument(
        '-ss',
        '--source_scene',
        type=str,
        default='./datasets/Multi_Illum_Invariance/test/everett_dining1')
    parser.add_argument(
        '-ts',
        '--target_scene',
        type=str,
        default='./datasets/Multi_Illum_Invariance/test/everett_living4')

    parser.add_argument("--cropx", type=int, default=512)
    parser.add_argument("--cropy", type=int, default=256)

    parser.add_argument("--cropsz", type=int, default=512)
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--chromesz", type=int, default=64)
    parser.add_argument("--fmaps1", type=int, default=6)
    parser.add_argument("--models", type=str)

    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument(
        '--illum_weights',
        type=str)
    parser.add_argument('--illum_model', type=str, default='unet')

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shift_range", action='store_true')
    parser.add_argument("--reference_probe", action='store_true')
    parser.add_argument("--baseline", action='store_true')
    parser.add_argument("--crop_images", action='store_true')
    parser.add_argument("--mask_probes", action='store_true')
    parser.add_argument("--non_light_loss", action='store_true')
    parser.add_argument("--swap_channels", action='store_true')

    opts = parser.parse_args()

    main(opts)
