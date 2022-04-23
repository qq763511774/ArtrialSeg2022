import os
import AtriaSeg2018
import SimpleITK as sitk
from skimage import measure as skim
import numpy as np

data_dir = '/data/AtriaSeg2018/Merged'


mask_file_name = 'mask.nrrd'


def load_nrrd(full_path_filename):

    data = sitk.ReadImage(full_path_filename)
    data = sitk.Cast(sitk.RescaleIntensity(data), sitk.sitkUInt8)
    data = sitk.GetArrayFromImage(data)

    data[data.nonzero()] = 1

    return(data)


folders = os.listdir(data_dir)
folders.sort()

for dir in folders:
    print(dir)
    mask = load_nrrd(data_dir + '/' + dir + '/' + mask_file_name)
    labels, num = skim.label(mask, connectivity=2, return_num=True)
    if num > 1:
        predict_np = np.zeros(mask.shape, dtype=np.uint8)
        props = skim.regionprops(labels)
        for k in range(len(props)):
            r = float(props[k].area) / float(len(mask.nonzero()[0]))
            print(r)
            if r >= 0.005:
                idx = np.argwhere(labels == k + 1)
                predict_np[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

        image = predict_np
        sitk.WriteImage(sitk.GetImageFromArray(image), data_dir + '/' + dir + '/' + mask_file_name)



