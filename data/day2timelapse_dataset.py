"""
day2timelapse_dataset.py:
Dataset loader for day2timelapse. It loads images belonging to Waymo
Day/Dusk/Dawn/Night splits and it applies a tone mapping operator to
the "Day" ones in order to drive learning with CoMoGAN.
It has support for custom options in DatasetOptions.
"""

import os.path

import numpy as np
import math
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import random
from torchvision.transforms import ToTensor
import torch
import munch


def DatasetOptions():
    do = munch.Munch()
    do.num_threads = 4
    do.batch_size = 1
    do.preprocess = 'none'
    do.max_dataset_size = float('inf')
    do.no_flip = False
    do.serial_batches = False
    return do


class Day2TimelapseDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_day = os.path.join(opt.dataroot, 'sunny', 'Day')
        self.dir_dusk = os.path.join(opt.dataroot, 'sunny', 'Dawn', 'Dusk')
        self.dir_night = os.path.join(opt.dataroot, 'sunny', 'Night')
        self.A_paths = [os.path.join(self.dir_day, x) for x in os.listdir(self.dir_day)]   # load images from '/path/to/data/trainA'
        self.B_paths = [os.path.join(self.dir_dusk, x) for x in os.listdir(self.dir_dusk)]    # load images from '/path/to/data/trainB'
        self.B_paths += [os.path.join(self.dir_night, x) for x in os.listdir(self.dir_night)]    # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.A_paths.sort()
        self.B_paths.sort()
        self.transform_A = get_transform(self.opt, grayscale=(opt.input_nc == 1), convert=False)
        self.transform_B = get_transform(self.opt, grayscale=(opt.output_nc == 1), convert=False)

        self.__tonemapping = torch.tensor(np.loadtxt('./data/daytime_model_lut.csv', delimiter=','),
                                   dtype=torch.float32)

        self.__xyz_matrix = torch.tensor([[0.5149,   0.3244,   0.1607],
                                   [0.2654,   0.6704,   0.0642],
                                   [0.0248,   0.1248,   0.8504]])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        # Define continuity normalization
        A = ToTensor()(A)
        B = ToTensor()(B)

        phi = random.random() * 2 * math.pi
        continuity_sin = math.sin(phi)
        cos_phi = math.cos(phi)

        A_cont = self.__apply_colormap(A, cos_phi, continuity_sin)

        phi_prime = random.random() * 2 * math.pi
        sin_phi_prime = math.sin(phi_prime)
        cos_phi_prime = math.cos(phi_prime)

        A_cont_compare = self.__apply_colormap(A, cos_phi_prime, sin_phi_prime)

        # Normalization between -1 and 1
        A = (A * 2) - 1
        B = (B * 2) - 1
        A_cont = (A_cont * 2) - 1
        A_cont_compare = (A_cont_compare * 2) - 1


        return {'A': A, 'B': B, 'A_cont': A_cont, 'A_paths': A_path, 'B_paths': B_path, 'cos_phi': float(cos_phi),
                'sin_phi': float(continuity_sin), 'sin_phi_prime': float(sin_phi_prime),
                'cos_phi_prime': float(cos_phi_prime), 'A_cont_compare': A_cont_compare, 'phi': phi,
                'phi_prime': phi_prime,}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def __apply_colormap(self, im, cos_phi, sin_phi, eps = 1e-8):
        size_0, size_1, size_2 = im.size()
        cos_phi_norm = 1 - (cos_phi + 1) / 2 # 0 in 0, 1 in pi
        im_buf = im.permute(1, 2, 0).view(-1, 3)
        im_buf = torch.matmul(im_buf, self.__xyz_matrix)

        X = im_buf[:, 0] + eps
        Y = im_buf[:, 1]
        Z = im_buf[:, 2]

        V = Y * (1.33 * (1 + (Y + Z) / X) - 1.68)

        tmp_index_lower = int(cos_phi_norm * self.__tonemapping.size(0))

        if tmp_index_lower < self.__tonemapping.size(0) - 1:
            tmp_index_higher = tmp_index_lower + 1
        else:
            tmp_index_higher = tmp_index_lower
        interp_index = cos_phi_norm * self.__tonemapping.size(0) - tmp_index_lower
        try:
            color_lower = self.__tonemapping[tmp_index_lower, :3]
        except IndexError:
            color_lower = self.__tonemapping[-2, :3]
        try:
            color_higher = self.__tonemapping[tmp_index_higher, :3]
        except IndexError:
            color_higher = self.__tonemapping[-2, :3]
        color = color_lower * (1 - interp_index) + color_higher * interp_index


        if sin_phi >= 0:
            # red shift
            corr = torch.tensor([0.1, 0, 0.1]) * sin_phi # old one was 0.03
        if sin_phi < 0:
            # purple shift
            corr = torch.tensor([0.1, 0, 0]) * (- sin_phi)

        color += corr
        im_degree = V.unsqueeze(1) * torch.matmul(color, self.__xyz_matrix)
        im_degree = torch.matmul(im_degree, self.__xyz_matrix.inverse()).view(size_1, size_2, size_0).permute(2, 0, 1)
        im_final = im_degree * cos_phi_norm + im * (1 - cos_phi_norm) + corr.unsqueeze(-1).unsqueeze(-1).repeat(1, im_degree.size(1), im_degree.size(2))

        im_final = im_final.clamp(0, 1)
        return im_final
