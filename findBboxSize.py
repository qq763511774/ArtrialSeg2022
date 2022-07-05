import SimpleITK as sitk
import numpy as np
import torch.utils.data as tdata
import AtriaSeg2022

data_dir = '/home/bob/Datasets/AtriaSeg/task2/train_data'

def get_item(mask_path):
    mask = mask_path

    # load mask data (binary)
    mask_data = load_niigz(mask, False, False, True)
    #            print(mask_data.shape)
    xborder, yborder, zborder = findBorder(mask_data)
    return xborder, yborder, zborder

def findBorder(volume):
    x,y,z = volume.nonzero()
    l = [min(x),max(x),min(y),max(y),min(z),max(z)]
    return [l[1]-l[0],l[3]-l[2],l[5]-l[4]]

#  1024,1024,96
def load_niigz(full_path_filename, is_origin=True, is_downsample=False, is_binary=False):
    volume_data = sitk.ReadImage(full_path_filename)
    volume_data = sitk.GetArrayFromImage(volume_data)
    print(full_path_filename)

    # exchange the first and last axis
    # (88, 576, 576) -> (576, 576, 88), (88, 640, 640) -> (640, 640, 88)
    volume_data = volume_data.swapaxes(0, 2)

    if is_origin:
        if is_binary:
            volume_data[volume_data.nonzero()] = 1
        return volume_data

    # 640,576,1024,922; 55,44
    if volume_data.shape[0] == 576:
        volume_data = np.pad(volume_data, ((224,224),(224,224),(0,0)),'constant')

    if volume_data.shape[0] == 640:
        volume_data = np.pad(volume_data, ((192,192),(192,192),(0,0)),'constant')
    
    if volume_data.shape[0] == 922:
        volume_data = np.pad(volume_data, ((51,51),(51,51),(0,0)),'constant')

    if volume_data.shape[2] == 44:
        volume_data = np.pad(volume_data, ((0, 0), (0, 0), (52,52)), 'constant')

    if volume_data.shape[2] == 55:
        volume_data = np.pad(volume_data, ((0, 0), (0, 0), (42, 42)), 'constant')

    if is_downsample:
        volume_data = volume_data[::4, ::4, ::2]

    # make the mask binary
    if is_binary:
        volume_data[volume_data.nonzero()] = 1

    return volume_data


if __name__ == '__main__':
    volume_file_name = 'enhanced_resampled.nii.gz'
    mask_file_name = 'atriumSegImgMO_resampled.nii.gz'
    volumes, masks = [], []
    All_paths = np.array(AtriaSeg2022.get_all_paths(data_dir))
    # paths is a list of full path
    cnt = 0
    for path in All_paths:
        volume = path + '/' + volume_file_name
        mask = path + '/' + mask_file_name
        volumes.append(volume)
        masks.append(mask)
        cnt += 1
        if(cnt%5 == 0):
            print(cnt)
    xlenMax, ylenMax, zlenMax = 0, 0, 0
    cnt = 0
    for mask in masks:
        l = get_item(mask)
        if(xlenMax < l[0]):
            xlenMax = l[0]
        if(ylenMax < l[1]):
            ylenMax = l[1]
        if(zlenMax < l[2]):
            zlenMax = l[2]
        cnt += 1
        if(cnt%5 == 0):
            print(cnt)
            print('\n',xlenMax,ylenMax,zlenMax)
    print(xlenMax,ylenMax,zlenMax)