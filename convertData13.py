import os
import AtriaSeg2018
import numpy as np
import SimpleITK as sitk

volume_file_name = 'image.mhd'
mask_file_name = 'gt_noclip.mhd'
#mask_file_name = 'gt_binary.mhd'

data_dir = 'AtriaSeg2013/Train'
out_dir_pre = 'AtriaSeg2013/TrainSub/b'

dir_path = os.listdir(data_dir)
dir_path.sort()

bbox_size = (240, 160, 96)

def check_dir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)

maxX = []
maxY = []
maxZ = []

for i, dir in enumerate(dir_path):
    if not dir.startswith('.'):

        print(dir)

        image = AtriaSeg2018.load_mhd(data_dir + '/' + dir + '/' + volume_file_name, False)
        mask = AtriaSeg2018.load_mhd(data_dir + '/' + dir + '/' + mask_file_name, True)

        print(mask.shape)

        midx, midy, midz = AtriaSeg2018.find_centers(mask)

        bbminx = int(midx - bbox_size[0] // 2)
        bbminy = int(midy - bbox_size[1] // 2)
        bbminz = int(midz - bbox_size[2] // 2)

        # crop data with bbox_size
        image = image[:, bbminy:bbminy + bbox_size[1], bbminz:bbminz + bbox_size[2]]
        mask = mask[:, bbminy:bbminy + bbox_size[1], bbminz:bbminz + bbox_size[2]]

        shape = mask.shape

        if bbminx < 0 and bbminx + bbox_size[0] < shape[0]:
            image = image[0:bbminx + bbox_size[0], :, :]
            mask = mask[0:bbminx + bbox_size[0], :, :]
            image = np.pad(image, ((abs(bbminx), 0), (0, 0), (0, 0)), 'constant')
            mask = np.pad(mask, ((abs(bbminx), 0), (0, 0), (0, 0)), 'constant')
        elif bbminx >= 0 and bbminx + bbox_size[0] < shape[0]:
            image = image[bbminx:bbminx + bbox_size[0], :, :]
            mask = mask[bbminx:bbminx + bbox_size[0], :, :]
        elif bbminx > 0 and bbminx + bbox_size[0] >= shape[0]:
            image = image[bbminx:, :, :]
            mask = mask[bbminx:, :, :]
            image = np.pad(image, ((0, bbminx + bbox_size[0] - shape[0]), (0, 0), (0, 0)), 'constant')
            mask = np.pad(mask, ((0, bbminx + bbox_size[0] - shape[0]), (0, 0), (0, 0)), 'constant')
        elif bbminx < 0 and bbminx + bbox_size[0] >= shape[0]:
            image = np.pad(image, ((abs(bbminx), bbox_size[0] - abs(bbminx) - shape[0]), (0, 0), (0, 0)), 'constant')
            mask = np.pad(mask, ((abs(bbminx), bbox_size[0] - abs(bbminx) - shape[0]), (0, 0), (0, 0)), 'constant')

        print(mask.shape)

        image = image.swapaxes(0, 2)
        check_dir(out_dir_pre + '{:0>3}'.format(i + 1))
        sitk.WriteImage(sitk.GetImageFromArray(image), out_dir_pre + '{:0>3}'.format(i + 1) + '/' + 'image.nrrd')
        mask = mask.swapaxes(0, 2)
        sitk.WriteImage(sitk.GetImageFromArray(mask), out_dir_pre + '{:0>3}'.format(i + 1) + '/' + 'mask.nrrd')


