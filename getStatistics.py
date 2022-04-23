import AtriaSeg2018
import numpy as np
import scipy.ndimage
data_dir = 'Training Set 100'

volume_file_name = 'lgemri.nrrd'
mask_file_name = 'laendo.nrrd'

bbox_size = (256, 160, 96)

All_paths = AtriaSeg2018.get_all_paths(data_dir)

fold_K = 4
test_paths = All_paths[fold_K * 20:fold_K * 20 + 20]
train_paths = list(set(All_paths).difference(test_paths))

target_mean = 0

maxX = []
maxY = []
maxZ = []

"""
# paths is a list of full path
for i, path in enumerate(All_paths):
    print(i)
    volume = AtriaSeg2018.load_nrrd(path + '/' + volume_file_name, is_binary=False)
    mask = AtriaSeg2018.load_nrrd(path + '/' + mask_file_name, is_binary=True)

    # crop (640, 640, 88) to (576, 576, 88)
    if mask.shape == (640, 640, 88):
        #        volume = volume[32:608, 32:608, :]
        mask = mask[32:608, 32:608, :]

    # pad (576, 576, 88) to (576, 576, 96)
    #    volume = np.pad(volume, ((0, 0), (0, 0), (4, 4)), 'constant')
    mask = np.pad(mask, ((0, 0), (0, 0), (4, 4)), 'constant')



    x, y, z = mask.nonzero()
    xmax = np.max(x)
    xmin = np.min(x)
    ymax = np.max(y)
    ymin = np.min(y)
    zmax = np.max(z)
    zmin = np.min(z)

    print(xmin, xmax, ymin, ymax, zmin, zmax)
    # print(xmax - xmin, ymax - ymin, zmax - zmin)
    maxX.append(xmax - xmin)
    maxY.append(ymax - ymin)
    maxZ.append(zmax - zmin)

    midx, midy, midz = AtriaSeg2018.find_centers(mask)

    bbminx = int(midx - bbox_size[0] // 2)
    bbminy = int(midy - bbox_size[1] // 2)
    # bminz = int(midz - bbox_size[2] // 2)

    print(bbminx, bbminx+bbox_size[0], bbminy, bbminy+bbox_size[1], 0, bbox_size[2])

    x = np.sum(mask)
    # crop data with bbox_size
#    volume = volume[bbminx:bbminx + bbox_size[0], bbminy:bbminy + bbox_size[1], :]
    mask = mask[bbminx:bbminx + bbox_size[0], bbminy:bbminy + bbox_size[1], :]
    y = np.sum(mask)

    if x != y:
        print(x, y)

    # print(volume.shape, mask.shape)

    target_mean += np.mean(mask)
 #   print(np.mean(mask))


target_mean /= len(All_paths)

print(target_mean)

print(np.max(maxX), np.max(maxY), np.max(maxZ))
"""

# paths is a list of full path
for i, path in enumerate(test_paths):
    print(i)
    volume = AtriaSeg2018.load_nrrd(path + '/' + volume_file_name, is_binary=False)
    mask = AtriaSeg2018.load_nrrd(path + '/' + mask_file_name, is_binary=True)

    midx, midy, midz = AtriaSeg2018.find_centers(mask)

    bbminx = int(midx - bbox_size[0] // 2)
    bbminy = int(midy - bbox_size[1] // 2)

    #print(bbminx, bbminx + bbox_size[0], bbminy, bbminy + bbox_size[1], 0, bbox_size[2])

    # crop data with bbox_size
    volume = volume[bbminx:bbminx + bbox_size[0], bbminy:bbminy + bbox_size[1], :]
    mask = mask[bbminx:bbminx + bbox_size[0], bbminy:bbminy + bbox_size[1], :]

    mean = np.mean(volume)
    std = np.std(volume)
    volume = (volume - mean) / std

#    scipy.misc.imsave('Images/Test/t-{}.jpg'.format(i),
#                      volume[:, :, bbox_size[2] // 2].T / np.max(volume) + 0.5 * mask[:, :, bbox_size[2] // 2].T)

#    volume, mask_data = AtriaSeg2018.random_rotation_3d(volume, mask, (5, 5, 10))
#    volume, mask_data = AtriaSeg2018.random_flip_3d(volume, mask)
#    volume, mask = AtriaSeg2018.elastic_deformation(volume, mask, mask.shape[0] * 3, mask.shape[0] * 0.05)

    volume = AtriaSeg2018.equalize_adapthist_3d(volume/np.max(volume))

    mean = np.mean(volume)
    std = np.std(volume)
    volume = (volume - mean) / std

    scipy.misc.imsave('Images/Test/t-{}.jpg'.format(i),
                      volume[:, :, volume.shape[2] // 2].T / np.max(volume) + 0.5 * mask[:, :, volume.shape[2]  // 2].T)

 #   print(midx, midy, midz)