from __future__ import division
import os
import torch.utils.data as data
import torch
import torch.nn as nn
import random
import SimpleITK as sitk
import numbers
import numpy as np
import scipy.ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter
from skimage.util import img_as_float, img_as_uint
from skimage.exposure import rescale_intensity


# bounding box size
bbox_size = (240, 160, 96)

NR_OF_GREY = 2 ** 14  # number of grayscale levels to use in CLAHE algorithm


def get_all_paths(root):
    paths = []
    dirs = os.listdir(root)
    dirs.sort()
    for folder in dirs:
        if not folder.startswith('.'):  # skip hidden folders
            path = root + '/' + folder
            paths.append(path)
    return paths


# refer to https://github.com/mattmacy/vnet.pytorch/blob/master/train.py
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)


#        m.bias.data.zero_()


# simple dataset wrapper
class Dataset(data.Dataset):
    # mode: 0->no down-sample & no crop, 1->down-sample & no crop, 2->no down-sample but crop
    def __init__(self, paths, mode, centers=[], transform=True, normalization=True):

        super(Dataset, self).__init__()
        self.mode = mode
        self.centers = centers

        # MRI volume and mask file name
        volume_file_name = 'enhanced.nii.gz'
        mask_file_name = 'atriumSegImgMO.nii.gz'

        # init folder names with empty set
        volumes, masks = [], []

        # paths is a list of full path
        for path in paths:
            volume = path + '/' + volume_file_name
            mask = path + '/' + mask_file_name
            volumes.append(volume)
            masks.append(mask)

        self.volumes = volumes
        self.masks = masks

        self.transform = transform
        self.normalization = normalization

    def __getitem__(self, index):

        volume = self.volumes[index]
        mask = self.masks[index]

        if self.mode == 1:  # down-sample
            # load volume data
            volume_data = load_niigz(volume, False, True, False) # origin, downsample, binary
            # load mask data (binary)
            mask_data = load_niigz(mask, False, True, True)

        elif self.mode == 2:  # no down-sample but crop
            # load volume data
            volume_data = load_niigz(volume, False, False, False)
            # load mask data (binary)
            mask_data = load_niigz(mask, False, False, True)
            #            print(mask_data.shape)

            # data augmentation: translation, rotation, scale
            if self.transform:
                # random scale
                if random.choice([True, False]):
                    volume_data, mask_data = random_scale_3d(volume_data, mask_data, (0.1, 0.1, 0.1))

                # find centers
                if self.centers:
                    midx, midy, midz = self.centers[index]
                else:
                    midx, midy, midz = find_centers(mask_data)

                x_offset = 0
                y_offset = 0

                # random translation
                if random.choice([True, False]):
                    x_offset = random.randint(-10, 10)
                    y_offset = random.randint(-10, 10)

                bbminx = int(midx - bbox_size[0] // 2)
                bbminy = int(midy - bbox_size[1] // 2)

                bbminx += x_offset
                bbminy += y_offset

                # crop data with bbox_size
                volume_data = volume_data[bbminx:bbminx + bbox_size[0], bbminy:bbminy + bbox_size[1], :]
                mask_data = mask_data[bbminx:bbminx + bbox_size[0], bbminy:bbminy + bbox_size[1], :]

                # random rotation
                if random.choice([True, False]):
                    volume_data, mask_data = random_rotation_3d(volume_data, mask_data, (5, 5, 10))

                # random flip
                if random.choice([True, False]):
                    volume_data, mask_data = random_flip_3d(volume_data, mask_data)

                # random elastic distortion
                if random.choice([True, False]):
                    volume_data, mask_data = elastic_deformation(volume_data, mask_data,
                                                                 volume_data.shape[0] * 3,
                                                                 volume_data.shape[0] * 0.05)
            else:

                if self.centers:
                    midx, midy, midz = self.centers[index]
                else:
                    midx, midy, midz = find_centers(mask_data)

                bbminx = int(midx - bbox_size[0] // 2)
                bbminy = int(midy - bbox_size[1] // 2)

                # crop data with bbox_size
                volume_data = volume_data[bbminx:bbminx + bbox_size[0], bbminy:bbminy + bbox_size[1], :]
                mask_data = mask_data[bbminx:bbminx + bbox_size[0], bbminy:bbminy + bbox_size[1], :]

            # sample-wise normalization & adaptive histogram normalization:
            if self.normalization:
                #                mean = np.mean(volume_data)
                #                std  =  np.std(volume_data)
                #                volume_data = (volume_data - mean) / std

                volume_data = equalize_adapthist_3d(volume_data / np.max(volume_data))

                mean = np.mean(volume_data)
                std = np.std(volume_data)
                volume_data = (volume_data - mean) / std

        else:
            # load volume data
            volume_data = load_niigz(volume, True, False, False)
            # load mask data (binary)
            mask_data = load_niigz(mask, True, False, True)

        # (144, 144, 48), (160,160,48), (144,144,70)
        volume_data = volume_data.reshape(1, volume_data.shape[0], volume_data.shape[1], volume_data.shape[2])
        volume_data = np.float32(volume_data)

        print(' volume data:',volume, '; shape', volume_data.shape)
        print(' mask data:',mask, '; shape', mask_data.shape)
        return volume_data, mask_data

    def __len__(self):
        return len(self.volumes)


# simple dataset wrapper
class DatasetTest(data.Dataset):
    # mode: 0->no down-sample & no crop, 1->down-sample & no crop, 2->no down-sample but crop
    def __init__(self, paths, mode, centers=[], normalization=True):

        super(DatasetTest, self).__init__()
        self.mode = mode
        self.centers = centers

        # MRI volume and mask file name
        volume_file_name = 'lgemri.nrrd'

        # init folder names with empty set
        volumes = []

        # paths is a list of full path
        for path in paths:
            volume = path + '/' + volume_file_name
            volumes.append(volume)

        self.volumes = volumes

        self.normalization = normalization

    def __getitem__(self, index):

        volume = self.volumes[index]

        if self.mode == 1:  # down-sample
            # load volume data
            volume_data = load_nrrd(volume, False, True, False)

        elif self.mode == 2:  # no down-sample but crop
            # load volume data
            volume_data = load_nrrd(volume, False, False, False)

            if self.centers:
                midx, midy, midz = self.centers[index]
            else:
                print('Centers are not found.')

            bbminx = int(midx - bbox_size[0] // 2)
            bbminy = int(midy - bbox_size[1] // 2)

            # crop data with bbox_size
            volume_data = volume_data[bbminx:bbminx + bbox_size[0], bbminy:bbminy + bbox_size[1], :]

            # sample-wise normalization & adaptive histogram normalization:
            if self.normalization:
                #                mean = np.mean(volume_data)
                #                std  =  np.std(volume_data)
                #                volume_data = (volume_data - mean) / std

                volume_data = equalize_adapthist_3d(volume_data / np.max(volume_data))

                mean = np.mean(volume_data)
                std = np.std(volume_data)
                volume_data = (volume_data - mean) / std

        else:
            # load volume data
            volume_data = load_nrrd(volume, True, False, False)

        #        print(volume_data.shape)
        volume_data = volume_data.reshape(1, volume_data.shape[0], volume_data.shape[1], volume_data.shape[2])
        volume_data = np.float32(volume_data)

        return volume_data

    def __len__(self):
        return len(self.volumes)


# simple dataset wrapper
class Dataset13(data.Dataset):
    def __init__(self, paths):

        super(Dataset13, self).__init__()

        # MRI volume and mask file name
        volume_file_name = 'image.nrrd'
        mask_file_name = 'mask.nrrd'

        # init folder names with empty set
        volumes, masks = [], []

        # paths is a list of full path
        for path in paths:
            volume = path + '/' + volume_file_name
            mask = path + '/' + mask_file_name
            volumes.append(volume)
            masks.append(mask)

        self.volumes = volumes
        self.masks = masks

    def __getitem__(self, index):

        volume = self.volumes[index]
        mask = self.masks[index]

        # load volume data
        volume_data = load_nrrd13(volume, False)
        # load mask data (binary)
        mask_data = load_nrrd13(mask, True)

        mean = np.mean(volume_data)
        std = np.std(volume_data)
        volume_data = (volume_data - mean) / std

        volume_data = volume_data.reshape(1, volume_data.shape[0], volume_data.shape[1], volume_data.shape[2])
        volume_data = np.float32(volume_data)

        return volume_data, mask_data

    def __len__(self):
        return len(self.volumes)


# this function loads .nrrd files into a 3D matrix and outputs it
# the input is the specified file path to the .nrrd file
def load_nrrd(full_path_filename, is_origin=True, is_downsample=False, is_binary=False):
    volume_data = sitk.ReadImage(full_path_filename)
    volume_data = sitk.GetArrayFromImage(volume_data)

    # exchange the first and last axis
    # (88, 576, 576) -> (576, 576, 88), (88, 640, 640) -> (640, 640, 88)
    volume_data = volume_data.swapaxes(0, 2)

    if is_origin:
        if is_binary:
            volume_data[volume_data.nonzero()] = 1
        return volume_data

    # crop (640, 640, 88) to (576, 576, 88)
    if volume_data.shape == (640, 640, 88):
        volume_data = volume_data[32:608, 32:608, :]

    # pad (576, 576, 88) to (576, 576, 96)
    volume_data = np.pad(volume_data, ((0, 0), (0, 0), (4, 4)), 'constant')

    # down-sample, ratio = (0.25, 0.25, 0.5)
    if is_downsample:
        volume_data = volume_data[::4, ::4, ::2]

    # make the mask binary
    if is_binary:
        volume_data[volume_data.nonzero()] = 1

    return volume_data


def load_mhd(full_path_filename, is_binary=False):

    volume_data = sitk.ReadImage(full_path_filename)
    spacing13 = volume_data.GetSpacing()
    volume_data = sitk.GetArrayFromImage(volume_data)

    volume_data = volume_data.swapaxes(1, 0)
    volume_data = volume_data.swapaxes(1, 2)
    volume_data = volume_data[::-1, :, :].copy()
    volume_data = volume_data[:, :, ::-1].copy()
    volume_data = volume_data.swapaxes(0, 2)

    sx18 = 0.625
    sy18 = 0.625
    sz18 = 1.25

    sx13 = spacing13[0]
    sy13 = spacing13[2]
    sz13 = spacing13[1]

    if is_binary:
        volume_data[volume_data.nonzero()] = 1
        volume_data = volume_data.astype(np.uint8)

    volume_data = scipy.ndimage.zoom(volume_data, (sx13 / sx18, sy13 / sy18, sz13 / sz18), order=2)

    return volume_data

# this function loads .nii files into a 3D matrix and outputs it
# the input is the specified file path to the .nii file
# low resolusion which is using downsample scale of 0.125,0.125,0.25
def load_niigz_low(full_path_filename, is_origin=True, is_downsample=False, is_binary=False):
    volume_data = sitk.ReadImage(full_path_filename)
    volume_data = sitk.GetArrayFromImage(volume_data)

    # exchange the first and last axis
    # (88, 576, 576) -> (576, 576, 88), (88, 640, 640) -> (640, 640, 88)
    volume_data = volume_data.swapaxes(0, 2)
    #        print(volume_data.shape)

    if is_origin:
        if is_binary:
            volume_data[volume_data.nonzero()] = 1
        return volume_data

    # crop (640, 640,) to (576, 576,)  origin:576 576 44, 576 576 88, 640 640 44, 640 640 88
    if volume_data.shape[0] == 640:
        volume_data = volume_data[32:608, 32:608, :]

    # pad (, 44) to (, 96)
    if volume_data.shape[2] == 44:
        volume_data = np.pad(volume_data, ((0, 0), (0, 0), (26, 26)), 'constant')

    # pad (, 88) to (, 96)
    if volume_data.shape[2] == 88:
        volume_data = np.pad(volume_data, ((0, 0), (0, 0), (4, 4)), 'constant')

    # down-sample, ratio = (0.125, 0.125, 0.25) 144 144 48, 72 72 24,
    if is_downsample:
        volume_data = volume_data[::8, ::8, ::4]

    # make the mask binary
    if is_binary:
        volume_data[volume_data.nonzero()] = 1

    return volume_data

def load_niigz(full_path_filename, is_origin=True, is_downsample=False, is_binary=False):
    volume_data = sitk.ReadImage(full_path_filename)
    volume_data = sitk.GetArrayFromImage(volume_data)

    # exchange the first and last axis
    # (88, 576, 576) -> (576, 576, 88), (88, 640, 640) -> (640, 640, 88)
    volume_data = volume_data.swapaxes(0, 2)
    #        print(volume_data.shape)

    if is_origin:
        if is_binary:
            volume_data[volume_data.nonzero()] = 1
        return volume_data

    # crop (640, 640,) to (576, 576,)  origin:576 576 44, 576 576 88, 640 640 44, 640 640 88
    if volume_data.shape[0] == 640:
        volume_data = volume_data[32:608, 32:608, :]

    # pad (, 44) to (, 96)
    if volume_data.shape[2] == 44:
        volume_data = np.pad(volume_data, ((0, 0), (0, 0), (26, 26)), 'constant')

    # pad (, 88) to (, 96)
    if volume_data.shape[2] == 88:
        volume_data = np.pad(volume_data, ((0, 0), (0, 0), (4, 4)), 'constant')

    # down-sample, ratio = (0.25, 0.25, 0.5) 144 144 48, 72 72 24,
    if is_downsample:
        volume_data = volume_data[::4, ::4, ::2]

    # make the mask binary
    if is_binary:
        volume_data[volume_data.nonzero()] = 1

    return volume_data

def load_nrrd13(full_path_filename, is_binary=False):

    volume_data = sitk.ReadImage(full_path_filename)
    volume_data = sitk.GetArrayFromImage(volume_data)

    volume_data = volume_data.swapaxes(0, 2)

    if is_binary:
        volume_data[volume_data.nonzero()] = 1
        volume_data = volume_data.astype(np.uint8)

    return volume_data
# compute the bounding box center of mask volume.

# mask: binary indicator
def find_centers(mask):
    x, y, z = mask.nonzero()

    midx = np.mean(x)
    midy = np.mean(y)
    midz = np.mean(z)

    return midx, midy, midz


# dice loss function
# predict: two volume indicate the probability to be foreground and background
# target: one volume of the ground-truth mask
def dice_loss_batch(predict, target):

    batch_size = predict.size()[0]

    dice_fg = 0
    dice_bg = 0

    for i in range(batch_size):
        # consider foreground
        predict_fg = predict[i, :, :, :, 1]
        predict_fg = predict_fg.view(predict_fg.numel())

        # cast int to float
        target_fg = target.float()
        target_fg = target_fg[i, :, :, :]
        target_fg = target_fg.view(target_fg.numel())

        smooth = 1.0

        intersection = torch.sum(predict_fg * target_fg, 0)
        union = torch.sum(predict_fg * predict_fg, 0) + torch.sum(target_fg * target_fg, 0)

        dice_fg += (2.0 * intersection + smooth) / (union + smooth)

        # consider foreground
        predict_bg = predict[i, :, :, :, 0]
        predict_bg = predict_bg.view(predict_bg.numel())

        # cast int to float
        target_bg = 1 - target.float()
        target_bg = target_bg[i, :, :, :]
        target_bg = target_bg.view(target_bg.numel())

        intersection = torch.sum(predict_bg * target_bg, 0)
        union = torch.sum(predict_bg * predict_bg, 0) + torch.sum(target_bg * target_bg, 0)

        dice_bg += (2.0 * intersection + smooth) / (union + smooth)

    dice_bg /= batch_size
    dice_fg /= batch_size

    return 2 - (dice_fg + dice_bg)


# dice loss function
# predict: two volume indicate the probability to be foreground and background
# target: one volume of the ground-truth mask
def dice_loss(predict, target):
    # consider foreground
    predict_fg = predict[:, 1]
    # cast int to float
    target_fg = target.float()

    smooth = 1.0

    intersection = torch.sum(predict_fg * target_fg, 0)
    union = torch.sum(predict_fg * predict_fg, 0) + torch.sum(target_fg * target_fg, 0)

    dice_fg = (2.0 * intersection + smooth) / (union + smooth)

    # consider background
    predict_bg = predict[:, 0]
    target_bg = 1 - target.float()

    intersection = torch.sum(predict_bg * target_bg, 0)
    union = torch.sum(predict_bg * predict_bg, 0) + torch.sum(target_bg * target_bg, 0)

    dice_bg = (2.0 * intersection + smooth) / (union + smooth)

    return 2 - (dice_fg + dice_bg)


def jaccard_loss(predict, target):

    # consider foreground
    predict_fg = predict[:, 1]
    # cast int to float
    target_fg = target.float()

    smooth = 1.0

    intersection = torch.sum(predict_fg * target_fg, 0)
    union = torch.sum(predict_fg * predict_fg, 0) + torch.sum(target_fg * target_fg, 0)

    jaccard_fg = (intersection + smooth) / (union - intersection + smooth)

    return 1 - jaccard_fg


def dice_loss_ohem_pixel(predict, target):

    # consider foreground
    predict_fg = predict[:, 1]
    # cast int to float
    target_fg = target.float()

    smooth = 1e-8

    intersections_fg = (predict_fg * target_fg)
    unions_fg = (predict_fg * predict_fg) + (target_fg * target_fg)

    loses_fg = 1 - (2.0 * intersections_fg + smooth) / (unions_fg + smooth)

    # consider background
    predict_bg = predict[:, 0]
    target_bg = 1 - target.float()

    intersections_bg = (predict_bg * target_bg)
    unions_bg = (predict_bg * predict_bg) + (target_bg * target_bg)

    loses_bg = 1 - (2.0 * intersections_bg + smooth) / (unions_bg + smooth)

    loses = loses_fg + loses_bg

    k = int(0.6 * target.numel())

    _, idxs = loses.topk(k)

    p = torch.randperm(idxs.numel())

    x = int(idxs.numel() * 0.3)

    idxs = idxs[p[0:x]]

    smooth = 1.0

    intersection_fg = torch.sum(intersections_fg[idxs], 0)
    union_fg = torch.sum(unions_fg[idxs], 0)

    dice_fg = (2.0 * intersection_fg + smooth) / (union_fg + smooth)

    intersection_bg = torch.sum(intersections_bg[idxs], 0)
    union_bg = torch.sum(unions_bg[idxs], 0)

    dice_bg = (2.0 * intersection_bg + smooth) / (union_bg + smooth)

    return 2 - (dice_fg + dice_bg)


def dice_loss_ohem(predict, target):

    batch_size = predict.size()[0]

    k = 1

    dices_fg = torch.zeros(batch_size)
    dices_bg = torch.zeros(batch_size)

    for i in range(batch_size):
        # consider foreground
        predict_fg = predict[i, :, :, :, 1]
        predict_fg = predict_fg.view(predict_fg.numel())

        # cast int to float
        target_fg = target.float()
        target_fg = target_fg[i, :, :, :]
        target_fg = target_fg.view(target_fg.numel())

        smooth = 1.0

        intersection = torch.sum(predict_fg * target_fg, 0)
        union = torch.sum(predict_fg * predict_fg, 0) + torch.sum(target_fg * target_fg, 0)

        dices_fg[i] = (2.0 * intersection + smooth) / (union + smooth)

        # consider foreground
        predict_bg = predict[i, :, :, :, 0]
        predict_bg = predict_bg.view(predict_bg.numel())

        # cast int to float
        target_bg = 1 - target.float()
        target_bg = target_bg[i, :, :, :]
        target_bg = target_bg.view(target_bg.numel())

        intersection = torch.sum(predict_bg * target_bg, 0)
        union = torch.sum(predict_bg * predict_bg, 0) + torch.sum(target_bg * target_bg, 0)

        dices_bg[i] = (2.0 * intersection + smooth) / (union + smooth)

    dices = dices_fg + dices_bg

    _, idxs = dices.topk(k)

    return 2 - dices[idxs].mean()


# dice function
# predict: two volume indicate the probability to be foreground and background
# target: one volume of the ground-truth mask
def dice_coeff(predict, target):
    batch_size = predict.size()[0]

    predict = predict.data.max(4)[1].float()
    target = target.float()

    dice = 0

    for i in range(batch_size):

        p = predict[i, :, :, :]
        p = p.view(p.numel())

        # cast int to float
        t = target[i, :, :, :]
        t = t.view(t.numel())

        intersection = torch.sum(p * t, 0)
        union = torch.sum(p * p, 0) + torch.sum(t * t, 0)

        dice += (2.0 * intersection) / union

    dice /= batch_size

    return dice


def dice_np(predict, target):

    predict = predict.reshape(-1)
    target = target.reshape(-1)
    intersection = np.sum(predict * target, 0)
    union = np.sum(predict * predict, 0) + np.sum(target * target, 0)
    return 2.0 * intersection / union


def elastic_deformation(volume, mask, alpha, sigma):
    """Elastic deformation of images as described in [Simard2003].
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

       based on https://gist.github.com/fmder/e28813c1e8721830ff9c
    """
    shape = volume.shape

    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))

    return map_coordinates(volume, indices, order=2).reshape(shape), \
           map_coordinates(mask, indices, order=1).reshape(shape)


def random_rotation_3d(volume, mask, max_angles):
    volume1 = volume
    mask1 = mask
    # rotate along z-axis
    angle = random.uniform(-max_angles[2], max_angles[2])
    volume2 = rotate(volume1, angle, order=2, mode='nearest', axes=(0, 1), reshape=False)
    mask2 = rotate(mask1, angle, order=1, mode='nearest', axes=(0, 1), reshape=False)

    # rotate along y-axis
    angle = random.uniform(-max_angles[1], max_angles[1])
    volume3 = rotate(volume2, angle, order=2, mode='nearest', axes=(0, 2), reshape=False)
    mask3 = rotate(mask2, angle, order=1, mode='nearest', axes=(0, 2), reshape=False)

    # rotate along x-axis
    angle = random.uniform(-max_angles[0], max_angles[0])
    volume_rot = rotate(volume3, angle, order=2, mode='nearest', axes=(1, 2), reshape=False)
    mask_rot = rotate(mask3, angle, order=1, mode='nearest', axes=(1, 2), reshape=False)

    return volume_rot, mask_rot


def random_scale_3d(volume, mask, max_scale_deltas):
    scalex = random.uniform(1 - max_scale_deltas[0], 1 + max_scale_deltas[0])
    scaley = random.uniform(1 - max_scale_deltas[1], 1 + max_scale_deltas[1])
    scalez = random.uniform(1 - max_scale_deltas[2], 1 + max_scale_deltas[2])

    volume_zoom = zoom(volume, (scalex, scaley, scalez), order=2)
    mask_zoom = zoom(mask, (scalex, scaley, scalez), order=1)

    if volume_zoom.shape[2] < bbox_size[2]:
        top = (bbox_size[2] - volume_zoom.shape[2]) // 2
        bot = (bbox_size[2] - volume_zoom.shape[2]) - top
        volume_zoom = np.pad(volume_zoom, ((0, 0), (0, 0), (bot, top)), 'constant')
        mask_zoom = np.pad(mask_zoom, ((0, 0), (0, 0), (bot, top)), 'constant')

    elif volume_zoom.shape[2] > bbox_size[2]:
        mid = volume_zoom.shape[2] // 2
        bot = mid - bbox_size[2] // 2
        top = bot + bbox_size[2]
        volume_zoom = volume_zoom[:, :, bot:top]
        mask_zoom = mask_zoom[:, :, bot:top]

    return volume_zoom, mask_zoom


def random_flip_3d(volume, mask):
    if random.choice([True, False]):
        volume = volume[::-1, :, :].copy()  # here must use copy(), otherwise error occurs
        mask = mask[::-1, :, :].copy()
    if random.choice([True, False]):
        volume = volume[:, ::-1, :].copy()
        mask = mask[:, ::-1, :].copy()
    if random.choice([True, False]):
        volume = volume[:, :, ::-1].copy()
        mask = mask[:, :, ::-1].copy()

    return volume, mask


"""
********************************************************
"""


def equalize_adapthist_3d(image, kernel_size=None,
                          clip_limit=0.01, nbins=256):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).
    An algorithm for local contrast enhancement, that uses histograms computed
    over different tile regions of the image. Local details can therefore be
    enhanced even in regions that are darker or lighter than most of the image.
    Parameters
    ----------
    image : (N1, ...,NN[, C]) ndarray
        Input image.
    kernel_size: integer or list-like, optional
        Defines the shape of contextual regions used in the algorithm. If
        iterable is passed, it must have the same number of elements as
        ``image.ndim`` (without color channel). If integer, it is broadcasted
        to each `image` dimension. By default, ``kernel_size`` is 1/8 of
        ``image`` height by 1/8 of its width.
    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast).
    nbins : int, optional
        Number of gray bins for histogram ("data range").
    Returns
    -------
    out : (N1, ...,NN[, C]) ndarray
        Equalized image.
    See Also
    --------
    equalize_hist, rescale_intensity
    Notes
    -----
    * For color images, the following steps are performed:
       - The image is converted to HSV color space
       - The CLAHE algorithm is run on the V (Value) channel
       - The image is converted back to RGB space and returned
    * For RGBA images, the original alpha channel is removed.
    References
    ----------
    .. [1] http://tog.acm.org/resources/GraphicsGems/
    .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE
    """
    image = img_as_uint(image)
    image = rescale_intensity(image, out_range=(0, NR_OF_GREY - 1))

    if kernel_size is None:
        kernel_size = tuple([image.shape[dim] // 8 for dim in range(image.ndim)])
    elif isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,) * image.ndim
    elif len(kernel_size) != image.ndim:
        ValueError('Incorrect value of `kernel_size`: {}'.format(kernel_size))

    kernel_size = [int(k) for k in kernel_size]

    image = _clahe(image, kernel_size, clip_limit * nbins, nbins)
    image = img_as_float(image)
    return rescale_intensity(image)


def _clahe(image, kernel_size, clip_limit, nbins=128):
    """Contrast Limited Adaptive Histogram Equalization.
    Parameters
    ----------
    image : (N1,...,NN) ndarray
        Input image.
    kernel_size: int or N-tuple of int
        Defines the shape of contextual regions used in the algorithm.
    clip_limit : float
        Normalized clipping limit (higher values give more contrast).
    nbins : int, optional
        Number of gray bins for histogram ("data range").
    Returns
    -------
    out : (N1,...,NN) ndarray
        Equalized image.
    The number of "effective" greylevels in the output image is set by `nbins`;
    selecting a small value (eg. 128) speeds up processing and still produce
    an output image of good quality. The output image will have the same
    minimum and maximum value as the input image. A clip limit smaller than 1
    results in standard (non-contrast limited) AHE.
    """

    if clip_limit == 1.0:
        return image  # is OK, immediately returns original image.

    ns = [int(np.ceil(image.shape[dim] / kernel_size[dim])) for dim in range(image.ndim)]

    steps = [int(np.floor(image.shape[dim] / ns[dim])) for dim in range(image.ndim)]

    bin_size = 1 + NR_OF_GREY // nbins
    lut = np.arange(NR_OF_GREY)
    lut //= bin_size

    map_array = np.zeros(tuple(ns) + (nbins,), dtype=int)

    # Calculate greylevel mappings for each contextual region

    for inds in np.ndindex(*ns):

        region = tuple([slice(inds[dim] * steps[dim], (inds[dim] + 1) * steps[dim]) for dim in range(image.ndim)])
        sub_img = image[region]

        if clip_limit > 0.0:  # Calculate actual cliplimit
            clim = int(clip_limit * sub_img.size / nbins)
            if clim < 1:
                clim = 1
        else:
            clim = NR_OF_GREY  # Large value, do not clip (AHE)

        hist = lut[sub_img.ravel()]
        hist = np.bincount(hist)
        hist = np.append(hist, np.zeros(nbins - hist.size, dtype=int))
        hist = clip_histogram(hist, clim)
        hist = map_histogram(hist, 0, NR_OF_GREY - 1, sub_img.size)
        map_array[inds] = hist

    # Interpolate greylevel mappings to get CLAHE image

    offsets = [0] * image.ndim
    lowers = [0] * image.ndim
    uppers = [0] * image.ndim
    starts = [0] * image.ndim
    prev_inds = [0] * image.ndim

    for inds in np.ndindex(*[ns[dim] + 1 for dim in range(image.ndim)]):

        for dim in range(image.ndim):
            if inds[dim] != prev_inds[dim]:
                starts[dim] += offsets[dim]

        for dim in range(image.ndim):
            if dim < image.ndim - 1:
                if inds[dim] != prev_inds[dim]:
                    starts[dim + 1] = 0

        prev_inds = inds[:]

        # modify edges to handle special cases
        for dim in range(image.ndim):
            if inds[dim] == 0:
                offsets[dim] = steps[dim] / 2.0
                lowers[dim] = 0
                uppers[dim] = 0
            elif inds[dim] == ns[dim]:
                offsets[dim] = steps[dim] / 2.0
                lowers[dim] = ns[dim] - 1
                uppers[dim] = ns[dim] - 1
            else:
                offsets[dim] = steps[dim]
                lowers[dim] = inds[dim] - 1
                uppers[dim] = inds[dim]

        maps = []
        for edge in np.ndindex(*([2] * image.ndim)):
            maps.append(map_array[tuple([[lowers, uppers][edge[dim]][dim] for dim in range(image.ndim)])])

        slices = [np.arange(starts[dim], starts[dim] + offsets[dim]) for dim in range(image.ndim)]

        interpolate(image, slices[::-1], maps, lut)

    return image


def clip_histogram(hist, clip_limit):
    """Perform clipping of the histogram and redistribution of bins.
    The histogram is clipped and the number of excess pixels is counted.
    Afterwards the excess pixels are equally redistributed across the
    whole histogram (providing the bin count is smaller than the cliplimit).
    Parameters
    ----------
    hist : ndarray
        Histogram array.
    clip_limit : int
        Maximum allowed bin count.
    Returns
    -------
    hist : ndarray
        Clipped histogram.
    """
    # calculate total number of excess pixels
    excess_mask = hist > clip_limit
    excess = hist[excess_mask]
    n_excess = excess.sum() - excess.size * clip_limit

    # Second part: clip histogram and redistribute excess pixels in each bin
    bin_incr = int(n_excess / hist.size)  # average binincrement
    upper = clip_limit - bin_incr  # Bins larger than upper set to cliplimit

    hist[excess_mask] = clip_limit

    low_mask = hist < upper
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr

    mid_mask = (hist >= upper) & (hist < clip_limit)
    mid = hist[mid_mask]
    n_excess -= mid.size * clip_limit - mid.sum()
    hist[mid_mask] = clip_limit

    prev_n_excess = n_excess

    while n_excess > 0:  # Redistribute remaining excess
        index = 0
        while n_excess > 0 and index < hist.size:
            under_mask = hist < 0
            step_size = int(hist[hist < clip_limit].size / n_excess)
            step_size = max(step_size, 1)
            indices = np.arange(index, hist.size, step_size)
            under_mask[indices] = True
            under_mask = (under_mask) & (hist < clip_limit)
            hist[under_mask] += 1
            n_excess -= under_mask.sum()
            index += 1
        # bail if we have not distributed any excess
        if prev_n_excess == n_excess:
            break
        prev_n_excess = n_excess

    return hist


def map_histogram(hist, min_val, max_val, n_pixels):
    """Calculate the equalized lookup table (mapping).
    It does so by cumulating the input histogram.
    Parameters
    ----------
    hist : ndarray
        Clipped histogram.
    min_val : int
        Minimum value for mapping.
    max_val : int
        Maximum value for mapping.
    n_pixels : int
        Number of pixels in the region.
    Returns
    -------
    out : ndarray
       Mapped intensity LUT.
    """
    out = np.cumsum(hist).astype(float)
    scale = ((float)(max_val - min_val)) / n_pixels
    out *= scale
    out += min_val
    out[out > max_val] = max_val
    return out.astype(int)


def interpolate(image, slices, maps, lut):
    """Find the new grayscale level for a region using bilinear interpolation.
    Parameters
    ----------
    image : ndarray
        Full image.
    slices : list of array-like
       Indices of the region.
    maps : list of ndarray
        Mappings of greylevels from histograms.
    lut : ndarray
        Maps grayscale levels in image to histogram levels.
    Returns
    -------
    out : ndarray
        Original image with the subregion replaced.
    Notes
    -----
    This function calculates the new greylevel assignments of pixels within
    a submatrix of the image. This is done by linear interpolation between
    2**image.ndim different adjacent mappings in order to eliminate boundary artifacts.
    """

    norm = np.product([slices[dim].size for dim in range(image.ndim)])  # Normalization factor

    # interpolation weight matrices
    coeffs = np.meshgrid(*tuple([np.arange(slices[dim].size) for dim in range(image.ndim)]), indexing='ij')
    coeffs = [coeff.transpose() for coeff in coeffs]

    inv_coeffs = [np.flip(coeffs[dim], axis=image.ndim - dim - 1) + 1 for dim in range(image.ndim)]

    region = tuple([slice(int(slices[dim][0]), int(slices[dim][-1] + 1)) for dim in range(image.ndim)][::-1])
    view = image[region]

    im_slice = lut[view]

    new = np.zeros_like(view, dtype=int)
    for iedge, edge in enumerate(np.ndindex(*([2] * image.ndim))):
        edge = edge[::-1]
        new += np.product([[inv_coeffs, coeffs][edge[dim]][dim] for dim in range(image.ndim)], 0) * maps[iedge][
            im_slice]

    new = (new / norm).astype(view.dtype)
    view[::] = new
    return image