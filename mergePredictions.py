import os
import SimpleITK as sitk
import numpy as np
from skimage import img_as_ubyte

data_dir = 'Testing Set 54'

folders = os.listdir(data_dir)
folders.sort()

data_dir_Li = '/data/AtriaSeg2018/HNet_prob'

data_dir_Huang = '/data/AtriaSeg2018/HN'

data_dir_Xia = '/data/AtriaSeg2018/VNET-COMBINE-ALL-MAPS/Full/Testing Set 54'

mask_dir = '/data/AtriaSeg2018/Merged'


def check_dir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)


def load_nrrd(full_path_filename):

    data = sitk.ReadImage(full_path_filename)
    data = sitk.Cast(sitk.RescaleIntensity(data), sitk.sitkUInt8)
    data = sitk.GetArrayFromImage(data)

    return(data)


for fold in folders:

    print(fold)

    path0_li = data_dir_Li + '/' + fold + '/' + 'p0.nrrd'
    path1_li = data_dir_Li + '/' + fold + '/' + 'p1.nrrd'

    p0_li = load_nrrd(path0_li) / 255
    p0_li[p0_li == 1] = 0
    image = img_as_ubyte(p0_li)
    sitk.WriteImage(sitk.GetImageFromArray(image), data_dir_Li + '/' + fold + '/' + 'p0-r.nrrd')

    p1_li = load_nrrd(path1_li) / 255

    path_huang = data_dir_Huang + '/' + fold + '/' + 'mask.nrrd'

    p1_huang = load_nrrd(path_huang) / 255
    p0_huang = (1 - p1_huang)

    path0_xia = data_dir_Xia + '/' + fold + '/' + 'p0.nrrd'
    path1_xia = data_dir_Xia + '/' + fold + '/' + 'p1.nrrd'

    p0_xia = load_nrrd(path0_xia) / 255
    p1_xia = load_nrrd(path1_xia) / 255

    check_dir(mask_dir + '/' + fold)

    p0 = (p0_li + p0_huang + p0_xia) / 3.0
    p1 = (p1_li + p1_huang + p1_xia) / 3.0

    p = np.stack((p0, p1), 0)

    p = np.argmax(p, 0)

    image_path = data_dir + '/' + fold + '/' + 'lgemri.nrrd'
    image = load_nrrd(image_path)
    sitk.WriteImage(sitk.GetImageFromArray(image), mask_dir + '/' + fold + '/image.nrrd')

    image = img_as_ubyte(p)
    sitk.WriteImage(sitk.GetImageFromArray(image), mask_dir + '/' + fold + '/mask.nrrd')
    p0 = (p0 - np.min(p0)) / (np.max(p0) - np.min(p0))
    image = img_as_ubyte(p0)
    sitk.WriteImage(sitk.GetImageFromArray(image), mask_dir + '/' + fold + '/p0.nrrd')
    p1 = (p1 - np.min(p1)) / (np.max(p1) - np.min(p1))
    image = img_as_ubyte(p1)
    sitk.WriteImage(sitk.GetImageFromArray(image), mask_dir + '/' + fold + '/p1.nrrd')