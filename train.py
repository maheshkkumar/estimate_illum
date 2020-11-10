import argparse
import os
import urllib
from time import time

import model

import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision
from multi_illum import MultiIllum
from utils import *

def autoexpose(I):
    """ Method to correct exposure
    """
    n = np.percentile(I[:,:,1], 90)
    if n > 0:
        I = I / n
    return I

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
    net = model.make_net_fully_convolutional(
                            chromesz=opts.chromesz,
                            fmaps1=opts.fmaps1,
                            xysize=opts.cropsz)
    net.to(device)

    # dataloader
    # dataset = MultiIllum(path, cropx, cropy, probe_size)
    trainset = MultiIllum(datapath=os.path.join(opts.path, 'train.txt'), cropx=opts.cropsz, cropy=opts.cropsz, probe_size=opts.chromesz)
    trainloader = DataLoader(trainset, batch_size=opts.batch_size, shuffle=True, num_workers=5)

    valset = MultiIllum(datapath=os.path.join(opts.path, 'val.txt'), cropx=opts.cropsz, cropy=opts.cropsz, probe_size=opts.chromesz)
    valloader = DataLoader(valset, batch_size=opts.batch_size, shuffle=False, num_workers=5)

    # optimizer and criterion
    optimizer = torch.optim.Adam(net.parameters(), lr=opts.learning_rate)
    criterion = torch.nn.L1Loss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    # tensorboard counter
    counter = 1

    # training
    for epoch in range(opts.epochs):
        start_time = time()
        for didx, data in enumerate(trainloader):
            img, probe = data
            img = img.to(device)
            probe = probe.to(device)

            optimizer.zero_grad()

            pred = net(img)
            pred = pred.reshape(pred.shape[0], 3, opts.chromesz, opts.chromesz)

            loss = criterion(pred, probe)
            loss.backward()
            optimizer.step()

            if (didx + 1) % opts.print_frequency == 0:
                end_time = time()
                time_taken = (end_time - start_time)
                print("Epoch: [{}]/[{}], Iteration: [{}]/[{}], Loss: {}, Time: {}s".format(epoch + 1, opts.epochs, didx + 1, len(trainloader), loss.item(), time_taken))
                writer.add_scalar('train/Loss', loss.item(), counter)
                writer.add_scalar('train/Learning Rate', optimizer.param_groups[0]['lr'], counter)

                if (didx + 1) % opts.save_images == 0:

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
        
        # save model
        model_path = os.path.join(opts.out, "models")
        check_folder(model_path)
        model_path = os.path.join(model_path, "epoch_{}.pth".format(epoch + 1))
        torch.save(net.state_dict(), model_path)

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
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument("--print_frequency", type=int, default=30)
    parser.add_argument("--save_images", type=int, default=180)

    opts = parser.parse_args()

    main(opts)