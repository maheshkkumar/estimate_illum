import os
from torch.utils.data import Dataset
from os.path import join as opj
from PIL import Image
from torchvision import transforms
import numpy as np
import torchvision.transforms.functional as TF

# TODO
# 1. Exposure correction
# 2. Image in [=-0.5, 0.5]
# 3. Resize image and probe
# 4. Mask the probes in the images

class MultiIllum(Dataset):
    def __init__(self, datapath: str, cropx: int, cropy: int, probe_size: int):
        self.datapath = datapath
        self.cropx = cropx
        self.cropy = cropy
        self.probe_size = probe_size
        self.filenames = self._get_files()
        self.img_transforms = transforms.Compose([transforms.RandomCrop((self.cropy, self.cropy)),
                                                  transforms.ToTensor()])
        self.probe_transforms = transforms.Compose([transforms.Resize(self.probe_size),
                                                    transforms.ToTensor()])

    def _get_files(self):
        if os.path.isfile(self.datapath):
            with open(self.datapath, 'r') as fp:
                folders = [folder.rstrip('\n') for folder in fp.readlines()]
        else:
            folders = [opj(self.datapath, folder) for folder in os.listdir(self.datapath)]
        print("Loading {} scene folders".format(len(folders)))

        filenames = []
        for folder in folders:
            filenames += [(opj(folder, img), opj(folder, 'probes', img.replace('mip2', 'chrome256'))) for img in os.listdir(folder) if img.endswith('mip2.jpg')]
        return filenames

    def __getitem__(self, index):
        img, probe = self.filenames[index]
        img = Image.open(img)
        probe = Image.open(probe)

        if np.random.random() > 0.5:
            img = TF.hflip(img)
            probe = TF.hflip(probe)

        img = self.img_transforms(img)
        probe = self.probe_transforms(probe)

        return img, probe

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # path = '/project/aksoy-lab/datasets/Multi_Illum_Invariance/test/'
    path = '/project/aksoy-lab/Mahesh/CMPT_726_Project/data/train.txt'
    cropx = 512
    cropy = 512
    probe_size = 64

    dataset = MultiIllum(path, cropx, cropy, probe_size)
    dataloader = DataLoader(dataset, batch_size=3)

    for data in dataloader:
        print(data[0].shape, data[1].shape)
        break