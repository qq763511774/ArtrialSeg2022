from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np

def resample_volume(volume_path, interpolator = sitk.sitkLinear, new_spacing = (0.625,0.625,2.0)):
    volume = sitk.ReadImage(volume_path.sitkFloat32) # read and cast to float32
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())

data_dir = '/home/bob/Datasets/AtriaSeg/task2/train_data/'
# data_dir = '/home/bob/Datasets/AtriaSeg/task2_val/test_'
volume_name = 'enhanced.nii.gz'
mask_name = 'atriumSegImgMO.nii.gz'

def resample_processing(iteration_):
    print(iteration_)
    new_volume_img = resample_volume(data_dir+'train_'+str(iteration_+1)+'/'+volume_name)
    new_mask_img = resample_volume(data_dir+'train_'+str(iteration_+1)+'/'+mask_name)
    sitk.WriteImage(new_volume_img, data_dir+'train_'+str(iteration_+1)+'/'+'enhanced_resampled.nii.gz')
    sitk.WriteImage(new_mask_img, data_dir+'train_'+str(iteration_+1)+'/'+'atriumSegImgMO_resampled.nii.gz')

with Pool() as pool:
    pool.map(resample_processing,range(130))