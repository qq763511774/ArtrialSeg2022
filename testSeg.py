import time
import os
import SimpleITK as sitk
import torch.utils.data as data
import torch
import torch.nn as nn
import numpy as np
import ResVNet
import AtriaSeg2018
from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")

# ----------for clusters---------
import argparse

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--gpu', default='0', type=str, metavar='N', help='use gpu')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# -------------------------------

device = torch.device("cuda")


def tic():
    return time.time()


def toc(start):
    stop = time.time()
    print('\nUsed {:.2f} s'.format(stop - start))


def check_dir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)


data_dir = 'Testing Set 54'

test_paths = AtriaSeg2018.get_all_paths(data_dir)

num_workers = 4

batch_size = 1  # this has to be 1 here

models1 = []
# model1 is used to roughly locate ROIs

for i in range(5):
    model1 = ResVNet.ResVNet8()
    model1 = nn.DataParallel(model1)
    model1.load_state_dict(torch.load('Models/vnet-mode1-fold{}-final.ckpt'.format(i)))
    models1.append(model1)


models2 = []
# model2 is used to precisely segment the target

for i in range(5):
    model2 = ResVNet.ResVNet()
    model2 = nn.DataParallel(model2)
    model2.load_state_dict(torch.load('Models/vnet-mode2-fold{}-final.ckpt'.format(i)))
    models2.append(model2)

for i in range(5):
    model2 = ResVNet.ResVNet()
    model2 = nn.DataParallel(model2)
    model2.load_state_dict(torch.load('Models/vnet-mode2-fold{}.ckpt'.format(i)))
    models2.append(model2)

for i in range(5):
    model2 = ResVNet.ResVNet()
    model2 = nn.DataParallel(model2)
    model2.load_state_dict(torch.load('Models/vnet-mode2-fold{}-nodeform-final.ckpt'.format(i)))
    models2.append(model2)

for i in range(5):
    model2 = ResVNet.ResVNet()
    model2 = nn.DataParallel(model2)
    model2.load_state_dict(torch.load('Models/vnet-mode2-fold{}-nodeform.ckpt'.format(i)))
    models2.append(model2)

dataset_mode = 1  # down-sample & no crop

fixed_size = (144, 144, 48)

predicted_centers = []

paths = test_paths
test_dataset = AtriaSeg2018.DatasetTest(paths, dataset_mode, normalization=True)
test_loader = data.DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

start = tic()

print('\nLocating region of interest...')

for i in range(5):
    models1[i].eval()

    for m in models1[i].modules():
        if isinstance(m, nn.BatchNorm3d):
            m.track_running_stats = False

with torch.no_grad():

    for i, data in enumerate(test_loader):

        print(test_paths[i])

        data = data.to(device)

        output_avg = torch.zeros(fixed_size[0]*fixed_size[1]*fixed_size[2], 2)

        output_avg = output_avg.to(device)

        # average over 5 models
        for j in range(5):
            output = models1[j](data)
            output_avg += output

        output_avg /= 5.0

        pred = output_avg.data.max(1)[1]

        p = pred.view(fixed_size[0], fixed_size[1], fixed_size[2]).cpu().numpy()
        o = data.view(fixed_size[0], fixed_size[1], fixed_size[2]).cpu().numpy()

        path = 'Predictions/Mode1/{}'.format(test_paths[i])
        check_dir(path)

        image = o
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = img_as_ubyte(image)
        image = image.swapaxes(0, 2)
        sitk.WriteImage(sitk.GetImageFromArray(image), path + '/image.nrrd')

        image = p.astype(np.uint8)
        image = image.swapaxes(0, 2)
        sitk.WriteImage(sitk.GetImageFromArray(image), path + '/mask.nrrd')

        pmidx, pmidy, pmidz = AtriaSeg2018.find_centers(p)

        predicted_centers.append((int(pmidx * 4.0), int(pmidy * 4.0), int(pmidz * 2.0)))

toc(start)

print('\nMode1 done.')

dataset_mode = 2  # no down-sample but crop

fixed_size = (240, 160, 96)

test_dataset2 = AtriaSeg2018.DatasetTest(paths, dataset_mode, predicted_centers, normalization=True)
test_loader2 = torch.utils.data.DataLoader(test_dataset2, num_workers=num_workers, batch_size=batch_size, shuffle=False)

start = tic()

print('\nSegmenting left artia from target region...')

prediction = []
foreground = []
background = []

for i in range(5):
    models2[i].eval()

    for m in models2[i].modules():
        if isinstance(m, nn.BatchNorm3d):
            m.track_running_stats = False

with torch.no_grad():

    for i, data in enumerate(test_loader2):

        print(test_paths[i])

        data = data.to(device)

        output_avg = torch.zeros(fixed_size[0]*fixed_size[1]*fixed_size[2], 2)

        output_avg = output_avg.to(device)

        # average over 5 models
        for j in range(len(models2)):
            output = models2[j](data)
            output_avg += output

        output_avg /= float(len(models2))

        pred = output_avg.data.max(1)[1]

        p = pred.view(fixed_size[0], fixed_size[1], fixed_size[2]).cpu().numpy()
        o = data.view(fixed_size[0], fixed_size[1], fixed_size[2]).cpu().numpy()

        path = 'Predictions/Mode2/{}'.format(test_paths[i])
        check_dir(path)

        image = o
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = img_as_ubyte(image)
        image = image.swapaxes(0, 2)
        sitk.WriteImage(sitk.GetImageFromArray(image), path + '/image.nrrd')

        predict_np = p.astype(np.uint8)
#
        p = predict_np.astype(np.uint8)
        prediction.append(p)

        image = predict_np.astype(np.uint8)
        image = image.swapaxes(0, 2)
        sitk.WriteImage(sitk.GetImageFromArray(image), path + '/mask.nrrd')

        f = output_avg[:, 1].view(fixed_size[0], fixed_size[1], fixed_size[2]).cpu().numpy()
        b = output_avg[:, 0].view(fixed_size[0], fixed_size[1], fixed_size[2]).cpu().numpy()

        image = img_as_ubyte(f)
        foreground.append(image)
        image = image.swapaxes(0, 2)
        sitk.WriteImage(sitk.GetImageFromArray(image), path + '/p1.nrrd')

        image = img_as_ubyte(b)
        background.append(image)
        image = image.swapaxes(0, 2)
        sitk.WriteImage(sitk.GetImageFromArray(image), path + '/p0.nrrd')

toc(start)
print('\nMode2 done.')


dataset_mode = 0

fixed_size = (576, 576, 88)

test_dataset3 = AtriaSeg2018.DatasetTest(paths, dataset_mode, normalization=False)
test_loader3 = torch.utils.data.DataLoader(test_dataset3, num_workers=num_workers, batch_size=batch_size, shuffle=False)

start = tic()
print('\nGenerating final masks...')

for i, data in enumerate(test_loader3):

    print(test_paths[i])

    x = data[0, 0, :, :, :].numpy()

    p0 = np.zeros(fixed_size)
    f0 = np.zeros(fixed_size)
    b0 = np.ones(fixed_size) * 255

    p = prediction[i]  # (260*140*96)
    f = foreground[i]
    b = background[i]

    midx, midy, midz = predicted_centers[i]

    bbminx = int(midx - p.shape[0] // 2)
    bbminy = int(midy - p.shape[1] // 2)

    p0[bbminx:bbminx + p.shape[0], bbminy:bbminy + p.shape[1], :] = p[:, :, 4:-4]
    f0[bbminx:bbminx + p.shape[0], bbminy:bbminy + p.shape[1], :] = f[:, :, 4:-4]
    b0[bbminx:bbminx + p.shape[0], bbminy:bbminy + p.shape[1], :] = b[:, :, 4:-4]

    if x.shape == (640, 640, 88):
        p = np.pad(p0, ((32, 32), (32, 32), (0, 0)), 'constant')
        f = np.pad(f0, ((32, 32), (32, 32), (0, 0)), 'constant')
        b = np.pad(b0, ((32, 32), (32, 32), (0, 0)), 'constant', constant_values=255)

    else:
        p = p0
        f = f0
        b = b0

    path = 'Predictions/Full/{}'.format(test_paths[i])
    check_dir(path)

    image = x
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = img_as_ubyte(image)
    image = image.swapaxes(0, 2)
    sitk.WriteImage(sitk.GetImageFromArray(image), path + '/image.nrrd')

    image = p.astype(np.uint8)
    image = image.swapaxes(0, 2)
    sitk.WriteImage(sitk.GetImageFromArray(image), path + '/mask.nrrd')

    image = img_as_ubyte(f)
    image = image.swapaxes(0, 2)
    sitk.WriteImage(sitk.GetImageFromArray(image), path + '/p1.nrrd')

    image = img_as_ubyte(b)
    image = image.swapaxes(0, 2)
    sitk.WriteImage(sitk.GetImageFromArray(image), path + '/p0.nrrd')

toc(start)
print('\nAll done.')
