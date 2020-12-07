import json
import os
from itertools import permutations
from os.path import join as opj

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms

# TODO
# 1. Exposure correction


class MultiIllum(Dataset):
    def __init__(
            self,
            datapath: str,
            cropx: int,
            cropy: int,
            probe_size: int,
            shift_range: bool = False,
            is_train: bool = True,
            crop_images: bool = False,
            mask_probes: bool = False,
            swap_channels: bool = False):
        self.datapath = datapath
        self.filenames = self._get_files()

        # image resolutions
        self.cropx = cropx
        self.cropy = cropy
        self.probe_size = probe_size

        # data configurations
        self.is_train = is_train  # adaptive settings between train and val
        # bool value to shift pixel range from [0, 1] to [-0.5, 0,5]
        self.shift_range = shift_range
        # bool value to choose between cropping and not cropping the training
        # data
        self.crop_images = crop_images
        # bool value to mask out the probes in the train, val and test data
        self.mask_probes = mask_probes
        self.swap_channels = swap_channels

        # channel swapping
        self.indices = list(permutations(range(3), 3))

        # crop images only for training
        if self.crop_images:
            self.img_transforms = transforms.Compose(
                [transforms.RandomCrop((self.cropy, self.cropy)), transforms.ToTensor()])
        else:
            self.img_transforms = transforms.Compose(
                [transforms.Resize((960, 1472)), transforms.ToTensor()])

        self.probe_transforms = transforms.Compose(
            [transforms.Resize(self.probe_size), transforms.ToTensor()])

    def _read_json(self, meta_data):
        with open(meta_data, 'r') as fp:
            meta_data = json.load(fp)

        chrome = {'x': int(meta_data['chrome']['bounding_box']['x'] / 4),
                  'y': int(meta_data['chrome']['bounding_box']['y'] / 4),
                  'w': int(meta_data['chrome']['bounding_box']['w'] / 4),
                  'h': int(meta_data['chrome']['bounding_box']['h'] / 4)
                  }

        gray = {'x': int(meta_data['gray']['bounding_box']['x'] / 4),
                'y': int(meta_data['gray']['bounding_box']['y'] / 4),
                'w': int(meta_data['gray']['bounding_box']['w'] / 4),
                'h': int(meta_data['gray']['bounding_box']['h'] / 4)
                }

        return chrome, gray

    def _get_files(self):
        if os.path.isfile(self.datapath):
            with open(self.datapath, 'r') as fp:
                folders = [folder.rstrip('\n') for folder in fp.readlines()]
        else:
            folders = [opj(self.datapath, folder)
                       for folder in os.listdir(self.datapath)]
        print("Loading {} scene folders".format(len(folders)))

        filenames = []
        for folder in folders:
            # scene images
            imgs = [img for img in os.listdir(
                folder) if img.endswith('mip2.jpg')]
            scene_imgs = []
            for img in imgs:
                # randomly picking a target image to relight the source image
                neg_imgs = [neg_im for neg_im in imgs if neg_im != img]
                target_img = np.random.choice(neg_imgs)
                scene_imgs.append(
                    (opj(
                        folder, img), opj(
                        folder, 'probes', img.replace(
                            'mip2', 'chrome256')), opj(
                        folder, target_img), opj(
                        folder, 'probes', target_img.replace(
                            'mip2', 'chrome256')), opj(
                                folder, 'meta.json')))
            filenames += scene_imgs
        return filenames

    def _apply_mask(self, img, chrome: dict, gray: dict):
        # applying mask over chrome sphere
        img[chrome['y']: chrome['y'] + chrome['h'],
            chrome['x']: chrome['x'] + chrome['w'], :] = 0

        # applying mask over gray sphere
        img[gray['y']: gray['y'] + gray['h'],
            gray['x']: gray['x'] + gray['w'], :] = 0

        return img

    def __getitem__(self, index):
        img, probe, target_img, target_probe, meta_data = self.filenames[index]

        # read image using skimage and mask out the probe if the bool value is
        # True and then convert it back to PIL.Image
        if self.mask_probes:
            chrome, gray = self._read_json(meta_data)
            img = Image.fromarray(
                self._apply_mask(
                    imread(img),
                    chrome,
                    gray).astype('uint8'),
                'RGB')
            target_img = Image.fromarray(
                self._apply_mask(
                    imread(target_img),
                    chrome,
                    gray).astype('uint8'),
                'RGB')
        else:
            img = Image.open(img)
            target_img = Image.open(target_img)

        probe = Image.open(probe)
        target_probe = Image.open(target_probe)

        # flip images only while training
        if self.is_train:
            if np.random.random() > 0.5:
                img = TF.hflip(img)
                target_img = TF.hflip(target_img)
                probe = TF.hflip(probe)
                target_probe = TF.hflip(target_probe)

            # augmentation to swap input channels
            if self.swap_channels:
                if np.random.random() > 0.5:
                    img = np.asarray(img)
                    img = Image.fromarray(
                        img[..., list(self.indices[np.random.randint(0, len(self.indices) - 1)])])

        img = self.img_transforms(img)
        target_img = self.img_transforms(target_img)
        probe = self.probe_transforms(probe)
        target_probe = self.probe_transforms(target_probe)

        # shift the range of images to be in [-0.5, 0.5]
        if self.shift_range:
            img -= 0.5
            target_img -= 0.5
            probe -= 0.5
            target_probe -= 0.5

        return img, probe, target_img, target_probe

    def __len__(self):
        return len(self.filenames)

class MultiIllumRelightingBaseline(Dataset):
    def __init__(
            self,
            datapath: str,
            cropx: int,
            cropy: int,
            probe_size: int,
            shift_range: bool = False,
            is_train: bool = True,
            crop_images: bool = False,
            mask_probes: bool = False,
            swap_channels: bool = False,
            random_relight: bool = False):
        self.datapath = datapath
        self.random_relight = random_relight
        self.filenames = self._get_files()

        # image resolutions
        self.cropx = cropx
        self.cropy = cropy
        self.probe_size = probe_size

        # data configurations
        self.is_train = is_train  # adaptive settings between train and val
        # bool value to shift pixel range from [0, 1] to [-0.5, 0,5]
        self.shift_range = shift_range
        # bool value to choose between cropping and not cropping the training
        # data
        self.crop_images = crop_images
        # bool value to mask out the probes in the train, val and test data
        self.mask_probes = mask_probes
        self.swap_channels = swap_channels

        # channel swapping
        self.indices = list(permutations(range(3), 3))

        # crop images only for training
        if self.crop_images:
            self.img_transforms = transforms.Compose(
                [transforms.RandomCrop((self.cropy, self.cropy)), transforms.ToTensor()])
        else:
            self.img_transforms = transforms.Compose(
                [transforms.Resize((960, 1472)), transforms.ToTensor()])

        self.probe_transforms = transforms.Compose(
            [transforms.Resize(self.probe_size), transforms.ToTensor()])

    def _read_json(self, meta_data):
        with open(meta_data, 'r') as fp:
            meta_data = json.load(fp)

        chrome = {'x': int(meta_data['chrome']['bounding_box']['x'] / 4),
                  'y': int(meta_data['chrome']['bounding_box']['y'] / 4),
                  'w': int(meta_data['chrome']['bounding_box']['w'] / 4),
                  'h': int(meta_data['chrome']['bounding_box']['h'] / 4)
                  }

        gray = {'x': int(meta_data['gray']['bounding_box']['x'] / 4),
                'y': int(meta_data['gray']['bounding_box']['y'] / 4),
                'w': int(meta_data['gray']['bounding_box']['w'] / 4),
                'h': int(meta_data['gray']['bounding_box']['h'] / 4)
                }

        return chrome, gray

    def _get_files(self):
        if os.path.isfile(self.datapath):
            with open(self.datapath, 'r') as fp:
                folders = [folder.rstrip('\n') for folder in fp.readlines()]
        else:
            folders = [opj(self.datapath, folder)
                       for folder in os.listdir(self.datapath)]
        print("Loading {} scene folders".format(len(folders)))

        filenames = []
        for folder in folders:
            # scene images
            if self.random_relight:
                imgs = [img for img in os.listdir(
                    folder) if img.endswith('mip2.jpg')]
                scene_imgs = []
                for img in imgs:
                    # randomly picking a target image to relight the source image
                    neg_imgs = [neg_im for neg_im in imgs if neg_im != img]
                    target_img = np.random.choice(neg_imgs)
                    scene_imgs.append((opj(folder, img), opj(folder, target_img), opj(folder, 'meta.json')))
                filenames += scene_imgs
            else:
                source_image = os.path.join(folder, 'dir_5_mip2.jpg')
                target_image = os.path.join(folder, 'dir_6_mip2.jpg')
                filenames.append((source_image, target_image, opj(folder, 'meta.json')))
        return filenames

    def _apply_mask(self, img, chrome: dict, gray: dict):
        # applying mask over chrome sphere
        img[chrome['y']: chrome['y'] + chrome['h'],
            chrome['x']: chrome['x'] + chrome['w'], :] = 0

        # applying mask over gray sphere
        img[gray['y']: gray['y'] + gray['h'],
            gray['x']: gray['x'] + gray['w'], :] = 0

        return img

    def __getitem__(self, index):
        img, target_img, meta_data = self.filenames[index]

        # read image using skimage and mask out the probe if the bool value is
        # True and then convert it back to PIL.Image
        if self.mask_probes:
            chrome, gray = self._read_json(meta_data)
            img = Image.fromarray(
                self._apply_mask(
                    imread(img),
                    chrome,
                    gray).astype('uint8'),
                'RGB')
            target_img = Image.fromarray(
                self._apply_mask(
                    imread(target_img),
                    chrome,
                    gray).astype('uint8'),
                'RGB')
        else:
            img = Image.open(img)
            target_img = Image.open(target_img)

        img = self.img_transforms(img)
        target_img = self.img_transforms(target_img)
        
        # shift the range of images to be in [-0.5, 0.5]
        if self.shift_range:
            img -= 0.5
            target_img -= 0.5
        
        return img, target_img

    def __len__(self):
        return len(self.filenames)


class TestMultiIllumRelighting(Dataset):
    def __init__(
            self,
            datapath: str,
            cropx: int,
            cropy: int,
            probe_size: int,
            shift_range: bool = False,
            is_train: bool = True,
            crop_images: bool = False,
            mask_probes: bool = False,
            swap_channels: bool = False):
        self.datapath = datapath
        self.filenames = self._get_files()

        # image resolutions
        self.cropx = cropx
        self.cropy = cropy
        self.probe_size = probe_size

        # data configurations
        self.is_train = is_train  # adaptive settings between train and val
        # bool value to shift pixel range from [0, 1] to [-0.5, 0,5]
        self.shift_range = shift_range
        # bool value to choose between cropping and not cropping the training
        # data
        self.crop_images = crop_images
        # bool value to mask out the probes in the train, val and test data
        self.mask_probes = mask_probes
        self.swap_channels = swap_channels

        # channel swapping
        self.indices = list(permutations(range(3), 3))

        # crop images only for training
        if self.crop_images:
            self.img_transforms = transforms.Compose(
                [transforms.RandomCrop((self.cropy, self.cropy)), transforms.ToTensor()])
        else:
            self.img_transforms = transforms.Compose(
                [transforms.Resize((960, 1472)), transforms.ToTensor()])

        self.probe_transforms = transforms.Compose(
            [transforms.Resize(self.probe_size), transforms.ToTensor()])

    def _read_json(self, meta_data):
        with open(meta_data, 'r') as fp:
            meta_data = json.load(fp)

        chrome = {'x': int(meta_data['chrome']['bounding_box']['x'] / 4),
                  'y': int(meta_data['chrome']['bounding_box']['y'] / 4),
                  'w': int(meta_data['chrome']['bounding_box']['w'] / 4),
                  'h': int(meta_data['chrome']['bounding_box']['h'] / 4)
                  }

        gray = {'x': int(meta_data['gray']['bounding_box']['x'] / 4),
                'y': int(meta_data['gray']['bounding_box']['y'] / 4),
                'w': int(meta_data['gray']['bounding_box']['w'] / 4),
                'h': int(meta_data['gray']['bounding_box']['h'] / 4)
                }

        return chrome, gray

    def _get_files(self):
        if os.path.isfile(self.datapath):
            with open(self.datapath, 'r') as fp:
                folders = [folder.rstrip('\n') for folder in fp.readlines()]
        else:
            folders = [opj(self.datapath, folder)
                       for folder in os.listdir(self.datapath)]
        print("Loading {} scene folders".format(len(folders)))

        filenames = []
        for folder in folders:
            # scene images
            imgs = [img for img in os.listdir(
                folder) if img.endswith('mip2.jpg')]
            scene_imgs = []
            for img in imgs:
                # randomly picking a target image to relight the source image
                scene_imgs.append(
                    (opj(
                        folder, img), opj(
                        folder, 'probes', img.replace(
                            'mip2', 'chrome256'))))
            filenames.append(
                [scene_imgs, opj(folder, 'meta.json'), folder.split(os.sep)[-1]])
        return filenames

    def _apply_mask(self, img, chrome: dict, gray: dict):
        # applying mask over chrome sphere
        img[chrome['y']: chrome['y'] + chrome['h'],
            chrome['x']: chrome['x'] + chrome['w'], :] = 0

        # applying mask over gray sphere
        img[gray['y']: gray['y'] + gray['h'],
            gray['x']: gray['x'] + gray['w'], :] = 0

        return img

    def __getitem__(self, index):
        scene_image_probes, meta_data, scene_name = self.filenames[index]

        # read image using skimage and mask out the probe if the bool value is
        # True and then convert it back to PIL.Image
        if self.mask_probes:
            chrome, gray = self._read_json(meta_data)
            images = [
                Image.fromarray(
                    self._apply_mask(
                        imread(
                            img[0]),
                        chrome,
                        gray).astype('uint8'),
                    'RGB') for img in scene_image_probes]
        else:
            images = [Image.open(img[0]) for img in scene_image_probes]

        probes = [Image.open(img[1]) for img in scene_image_probes]

        images = [self.img_transforms(img) for img in images]
        probes = [self.probe_transforms(probe) for probe in probes]

        # shift the range of images to be in [-0.5, 0.5]
        if self.shift_range:
            images = [img - 0.5 for img in images]
            probes = [probe - 0.5 for probe in probes]

        images = torch.stack(images)
        probes = torch.stack(probes)
        return images, probes, scene_name

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # path = '/project/aksoy-lab/datasets/Multi_Illum_Invariance/test/'
    path = '/project/aksoy-lab/Mahesh/CMPT_726_Project/data/test.txt'
    cropx = 512
    cropy = 512
    probe_size = 64

    # dataset = MultiIllum(path, cropx, cropy, probe_size)
    # dataloader = DataLoader(dataset, batch_size=3)

    # for data in dataloader:
    #     print(data[0].shape, data[1].shape, data[2].shape, data[3].shape)
    #     break

    # dataset = TestMultiIllumRelighting(path, cropx, cropy, probe_size)
    # dataloader = DataLoader(dataset, batch_size=1)

    # for data in dataloader:
    #     images, probes, scene = data
    #     print(images.shape, probes.shape, scene[0])
    #     break

    dataset = MultiIllumRelightingBaseline(path, cropx, cropy, probe_size, random_relight=False)
    dataloader = DataLoader(dataset, batch_size=1)

    for data in dataloader:
        source_img, target_img = data
        print(source_img.shape, target_img.shape)
        break
