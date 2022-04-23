import os
import AtriaSeg2018
import SimpleITK as sitk

data_dir1 = '/data/AtriaSeg2018/VNET-XIA/VNET-COMBINE-ALL/Full/Testing Set 54'

data_dir2 = 'Predictions/Full/Testing Set 54'

mask_file_name1 = 'laendo.nrrd'
mask_file_name2 = 'mask.nrrd'

def load_nrrd(full_path_filename):

    data = sitk.ReadImage(full_path_filename)
    data = sitk.Cast(sitk.RescaleIntensity(data), sitk.sitkUInt8)
    data = sitk.GetArrayFromImage(data)

    data[data.nonzero()] = 1

    return(data)


folders = os.listdir(data_dir1)
folders.sort()

dices = 0

for dir in folders:
    print(dir)
    mask1 = load_nrrd(data_dir1 + '/' + dir + '/' + mask_file_name1)
    mask2 = load_nrrd(data_dir2 + '/' + dir + '/' + mask_file_name2)

    dice = AtriaSeg2018.dice_np(mask1, mask2)

    dices += dice

    print(dice)

dices /= float(len(folders))
print('Average dice: {}'.format(dices))


