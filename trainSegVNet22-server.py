import os
import torch.utils.data as tdata
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.ndimage
import torch
import time
import random
import torch.nn.functional as F
import ResVNet
import AtriaSeg2022
import warnings
import sys
warnings.filterwarnings("ignore")

# ----------for clusters---------
import argparse

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('fold', type=int)
parser.add_argument('--gpu', default='0', type=str, metavar='N', help='use gpu')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# -------------------------------

use_cuda = True

device = torch.device("cuda" if use_cuda else "cpu")

# ----------Parameters-------------

# k-fold data set validation
fold_K = args.fold
# print('fold{}'.format(fold_K))

# number of processors for data processing
num_workers = 8

# training epochs 
epochs = 200

# mini-batch size, net1->4, net2->1, as described in the paper 
batch_size = 1

# train which net, 1->net1 (target localization), 2->net2 (atrial segmentation)
train_mode = 2

# random seed
seed = 0

# input data dir
data_dir = '/home/bob/Datasets/AtriaSeg/task2/train_data'

# model 1 dir
#pretrained_model_dir = './Out1/vnet-mode1-fold1.ckpt'

# output dir
output_dir = 'Out6-30'

# use data augmentation
is_use_daug = True

# intensity normalization & histogram equalization for input
is_use_normalization = True

is_output_train_images = True
is_output_test_images = True

# --------------------------------

# get all full paths of folders under data_dir
All_paths = AtriaSeg2022.get_all_paths(data_dir)

idxs = [82, 44, 19, 54, 62, 77, 11, 53, 8, 16, 23, 55, 98, 101, 51, 89, 105, 116, 83, 9,
        104, 125, 22, 30, 34, 117, 2, 90, 78, 84, 113, 118, 76, 66, 71, 107, 73, 80, 21, 
        121, 14, 86, 129, 127, 81, 18, 60, 24, 70, 25, 52, 68, 50, 95, 94, 57, 120, 35, 
        31, 96, 128, 15, 40, 111, 13, 103, 32, 92, 97, 75, 114, 69, 63, 12, 108, 87, 122,
        28, 33, 126, 72, 79, 29, 38, 74, 10, 20, 61, 39, 109, 64, 37, 56, 43, 124, 45,
        41, 48, 1, 7, 85, 6, 100, 27, 106, 93, 67, 17, 115, 26, 119, 65, 99, 3, 91, 42,
        0, 110, 49, 5, 47, 102, 123, 112, 58, 46, 59, 88, 4, 36]

sorted_paths = np.array(All_paths)[idxs]

test_idxs = []
test_idxs.append([2, 8, 13, 17, 20,  28, 32, 36, 41, 46,  50, 55, 61, 65, 73,  78, 84, 89, 90, 99,  101, 108, 111, 115, 120, 129]) # 1折的验证顺序
test_idxs.append([1, 7, 12, 16, 23, 26, 34, 38, 43, 48, 53, 58, 62, 66, 72, 76, 83, 86, 91, 96, 100, 107, 114, 117, 124, 126]) # 2折的验证顺序
test_idxs.append([3, 5, 11, 18, 24, 27, 31, 37, 44, 47, 52, 56, 63, 67, 71, 75, 81, 87, 94, 97, 103, 105, 112, 118, 122, 127]) # 3折的验证顺序
test_idxs.append([4, 6, 10, 15, 21, 29, 33, 39, 42, 49, 54, 57, 60, 69, 74, 77, 80, 85, 93, 98, 104, 106, 110, 119, 123, 125]) # 4折的验证顺序 
test_idxs.append([0, 9, 14, 19, 22, 25, 30, 35, 40, 45, 51, 59, 64, 68, 70, 79, 82, 88, 92, 95, 102, 109, 113, 116, 121, 128]) # 5折的验证顺序

test_paths = sorted_paths[test_idxs[fold_K]]
train_paths = list(set(sorted_paths).difference(test_paths))
train_paths.sort()

if train_mode == 1:
    fixed_size = (144, 144, 48)
else:
    fixed_size = (240, 160, 96)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def check_dir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(output_dir)


def _init_fn(worker_id):
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def tic():
    return time.time()


def toc(start):
    stop = time.time()
    print('\nUsed {:.2f} s\n'.format(stop - start))
    return stop - start


def train(epoch, model, train_loader, optimizer, fwriter):
 
    model.train()
    n_processed = 0
    n_train = len(train_loader.dataset)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.view(target.numel())

        output = model(data)

        loss = AtriaSeg2022.dice_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 2)
        target = target.view(-1, fixed_size[0], fixed_size[1], fixed_size[2])

        dice = AtriaSeg2022.dice_coeff(output, target)

        n_processed += len(data)

        partial_epoch = epoch + (batch_idx + 1) / len(train_loader)
        print('Train Epoch: {:5.2f} [{:2}/{:2} ({:3.0f}%)] \tLoss: {:.3f} \tDice: {:.3f}'.format(
            partial_epoch, n_processed, n_train, 100. * (batch_idx + 1) / len(train_loader),
            loss.item(), dice))

        fwriter.write('{},{},{}\n'.format(partial_epoch, loss.item(), dice))
        fwriter.flush()


def test(epoch, model, test_loader, fwriter):

    loss = 0
    dice = 0
    model.eval()
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm3d):
    #         m.track_running_stats = False

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            target = target.view(target.numel())

            output = model(data)

            loss += AtriaSeg2022.dice_loss(output, target).item()

            output = output.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 2)
            target = target.view(-1, fixed_size[0], fixed_size[1], fixed_size[2])

            dice += AtriaSeg2022.dice_coeff(output, target).item()

        loss /= len(test_loader)  # loss function already averages over batch size
        dice /= len(test_loader)
        print('\nTest set: Average Loss: {:.4f}, Average Dice: {:.4f}\n'.format(loss, dice))

    fwriter.write('{},{},{}\n'.format(epoch, loss, dice))
    fwriter.flush()

    return dice


def main():

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('Using', torch.cuda.device_count(), 'GPU(s).')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = ResVNet.ResVNetSE().cuda()

    check_dir(output_dir)

    fwriter_train = open(output_dir + '/train-mode{}-fold{}.csv'.format(train_mode, fold_K), 'w')
    fwriter_test = open(output_dir + '/test-mode{}-fold{}.csv'.format(train_mode, fold_K), 'w')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
#    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200])
#    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=1e-8)
#    optimizer = optim.RMSprop(model.parameters(), weight_decay=1e-8)

    #    print('Preparing training data...')
    train_dataset = AtriaSeg2022.Dataset(train_paths, train_mode, transform=is_use_daug,
                                         normalization=is_use_normalization)
    train_loader = tdata.DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                    worker_init_fn=_init_fn)

    #    print('Preparing testing data...')
    test_dataset = AtriaSeg2022.Dataset(test_paths, train_mode, transform=False,
                                        normalization=is_use_normalization)
    test_loader = tdata.DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False,
                                   worker_init_fn=_init_fn)

    max_dice = 0
    train_time, test_time = 0, 0

    for epoch in range(epochs):
        start = tic()
        train(epoch, model, train_loader, optimizer, fwriter_train)
        train_time += toc(start)
        start = tic()
        dice = test(epoch, model, test_loader, fwriter_test)
        if dice > max_dice:
            # Save the model checkpoint
            torch.save(model.state_dict(), output_dir + '/vnet-mode{}-fold{}.ckpt'.format(train_mode, fold_K))
            max_dice = dice
        test_time += toc(start)
#        scheduler.step()
        if epoch%10 == 9:
            torch.save(model.state_dict(), output_dir + '/vnet-mode{}-fold{}-autosave.ckpt'.format(train_mode, fold_K))
        print('Train time accumulated: {:.2f}s, Test time accumulated: {:.2f}s\n'.format(train_time, test_time))
    # Save the model checkpoint
    torch.save(model.state_dict(), output_dir + '/vnet-mode{}-fold{}-final.ckpt'.format(train_mode, fold_K))

    print('-------Max dice: {}--------'.format(max_dice))

if __name__ == '__main__':
    main()
