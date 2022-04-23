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
import AtriaSeg2018
import warnings
warnings.filterwarnings("ignore")

# ----------for clusters---------
import argparse

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--gpu', default='0', type=str, metavar='N', help='use gpu')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# -------------------------------

use_cuda = True

device = torch.device("cuda" if use_cuda else "cpu")

# ----------Parameters-------------

# k-fold data set validation
fold_K = 0 

# number of processors for data processing
num_workers = 8

# training epochs
epochs = 200

# mini-batch size, net1->4, net2->1, as described in the paper
batch_size = 4

# train which net, 1->net1 (target localization), 2->net2 (atrial segmentation)
train_mode = 1

# random seed
seed = 0

# input data dir
data_dir = 'Training Set 100'

# output dir
output_dir = 'Out'

# use dice loss or cross entropy loss
is_use_dice = True

# use data augmentation
is_use_daug = True

# intensity normalization & histogram equalization for input
is_use_normalization = True

is_output_train_images = False
is_output_test_images = False

# --------------------------------

# get all full paths of folders under data_dir
All_paths = AtriaSeg2018.get_all_paths(data_dir) # list

## 100个数据中用5折交叉验证进行训练后的结果验证，这里idxs应该是提前生成的随机序列
idxs = [31, 22, 50, 41, 39, 78, 37,  8, 16, 15, 28, 17, 38, 25, 45, 90, 96, 40,  2, 21, 35, 12,  5, 76, 73,
        26, 72, 91, 85, 54, 57, 30, 82, 53, 27, 18, 83, 66, 32, 86, 87, 84, 64,  9,  0, 68, 52, 62, 99, 24,
         3, 79, 14, 69, 59, 48, 34, 47,  6, 13, 80, 20, 46, 65,  4, 97, 56, 95, 92, 67, 44, 49, 58,  1, 42,
        11, 43, 71, 89, 29, 33, 81, 63, 98, 51, 75, 23, 93,  7, 60, 94, 55, 77, 88, 74, 10, 61, 36, 70, 19]

sorted_paths = np.array(All_paths)[idxs] #将all_paths按照idxs随机排序，生成sorted_paths

test_idxs = []
test_idxs.append([1, 6, 13, 19, 23, 29, 33, 35, 44, 48, 53, 55, 60, 66, 72, 77, 84, 86, 93, 98]) # 1折的验证顺序
test_idxs.append([3, 9, 10, 17, 20, 27, 30, 37, 40, 49, 52, 56, 63, 65, 70, 78, 82, 88, 94, 96]) # 2折的验证顺序
test_idxs.append([2, 7, 11, 16, 21, 26, 32, 38, 42, 45, 51, 59, 61, 69, 74, 75, 83, 85, 90, 97]) # 3折的验证顺序
test_idxs.append([0, 5, 14, 15, 22, 28, 34, 39, 43, 46, 50, 57, 62, 67, 71, 76, 80, 89, 92, 99]) # 4折的验证顺序 
test_idxs.append([4, 8, 12, 18, 24, 25, 31, 36, 41, 47, 54, 58, 64, 68, 73, 79, 81, 87, 91, 95]) # 5折的验证顺序

test_paths = sorted_paths[test_idxs[fold_K]] # 取出5折交叉验证中的第K折作为测试集
train_paths = list(set(sorted_paths).difference(test_paths)) # 其他的做训练集
train_paths.sort() # 把训练集再次排序

# k-fold data set
#test_paths = All_paths[fold_K * 20:fold_K * 20 + 20]
#train_paths = list(set(All_paths).difference(test_paths))
#train_paths.sort()

if train_mode == 1:
    fixed_size = (144, 144, 48)
else:
    fixed_size = (240, 160, 96)

torch.backends.cudnn.benchmark = True #网络结构与每个数据的维度一直保持不变的话，建议设true
torch.backends.cudnn.deterministic = True #会用更多时间，但是这会让喂相同data的时候


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


def train(epoch, model, train_loader, optimizer, weights, fwriter):

    model.train()
    n_processed = 0
    n_train = len(train_loader.dataset)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.view(target.numel())

        output = model(data)

        if is_use_dice:
#            loss = AtriaSeg2018.dice_loss(output, target)
             loss = AtriaSeg2018.dice_loss_ohem_pixel(output, target)
        else:
            loss = F.nll_loss(output, target.long(), weight=weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if is_output_train_images and epoch % 10 == 0:

            o = data.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 1)
            scipy.misc.imsave('Images/Train/o-{}-{}.jpg'.format(batch_idx, epoch),
                              o[0, :, :, fixed_size[2] // 2, 0].cpu().numpy().T)

            t = target.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 1).float() * o.max() / 3 + o
            scipy.misc.imsave('Images/Train/t-{}-{}.jpg'.format(batch_idx, epoch),
                              t[0, :, :, fixed_size[2] // 2, 0].cpu().numpy().T)

            pred = output.data.max(1)[1]  # get the index of the max log-probability
            p = pred.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 1).float() * o.max() / 3 + o
            scipy.misc.imsave('Images/Train/p-{}-{}.jpg'.format(batch_idx, epoch),
                              p[0, :, :, fixed_size[2] // 2, 0].cpu().numpy().T)
            """
            pred1 = output[:, 1]
            x = pred1.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 1)
            scipy.misc.imsave('Images/Train/p-{}-{}-1.jpg'.format(batch_idx, epoch),
                              x[0, :, :, fixed_size[2] // 2, 0].cpu().detach().numpy().T)

            pred2 = output[:, 0]
            x = pred2.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 1)
            scipy.misc.imsave('Images/Train/p-{}-{}-2.jpg'.format(batch_idx, epoch),
                              x[0, :, :, fixed_size[2] // 2, 0].cpu().detach().numpy().T)
            """
        output = output.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 2)
        target = target.view(-1, fixed_size[0], fixed_size[1], fixed_size[2])
        dice = AtriaSeg2018.dice_coeff(output, target)

        n_processed += len(data)

        partial_epoch = epoch + (batch_idx + 1) / len(train_loader)
        print('Train Epoch: {:5.2f} [{:2}/{:2} ({:3.0f}%)] \tLoss: {:.3f} \tDice: {:.3f}'.format(
            partial_epoch, n_processed, n_train, 100. * (batch_idx + 1) / len(train_loader),
            loss.item(), dice))

        fwriter.write('{},{},{}\n'.format(partial_epoch, loss.item(), dice))
        fwriter.flush()


def test(epoch, model, test_loader, weights, fwriter):

    loss = 0
    dice = 0
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.track_running_stats = False

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            target = target.view(target.numel())

            output = model(data)

            if is_use_dice:
#                loss += AtriaSeg2018.dice_loss(output, target).item()
                 loss += AtriaSeg2018.dice_loss_ohem_pixel(output, target).item()

            else:
                loss += F.nll_loss(output, target.long(), weight=weights)

            output = output.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 2)
            target = target.view(-1, fixed_size[0], fixed_size[1], fixed_size[2])
            dice += AtriaSeg2018.dice_coeff(output, target).item()

            if is_output_test_images and epoch % 5 == 0:

                o = data.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 1)
                scipy.misc.imsave('Images/Test/o-{}.jpg'.format(i),
                                  o[0, :, :, fixed_size[2] // 2, 0].cpu().numpy().T)

                t = target.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 1).float() * o.max() / 3 + o
                scipy.misc.imsave('Images/Test/t-{}.jpg'.format(i),
                                  t[0, :, :, fixed_size[2] // 2, 0].cpu().numpy().T)

                pred = output.data.max(1)[1]  # get the index of the max log-probability
                x = pred.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 1).float() * o.max() / 3 + o
                scipy.misc.imsave('Images/Test/p-{}-{}.jpg'.format(i, epoch),
                                  x[0, :, :, fixed_size[2] // 2, 0].cpu().numpy().T)
                """
                pred = out1.data.max(1)[1]  # get the index of the max log-probability
                x = pred.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 1).float()
                scipy.misc.imsave('Images/Test/p-{}-{}-1.jpg'.format(i, epoch),
                                  x[0, :, :, fixed_size[2] // 2, 0].cpu().numpy().T)

                pred = out2.data.max(1)[1]  # get the index of the max log-probability
                x = pred.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 1).float()
                scipy.misc.imsave('Images/Test/p-{}-{}-2.jpg'.format(i, epoch),
                                  x[0, :, :, fixed_size[2] // 2, 0].cpu().numpy().T)

                pred = out3.data.max(1)[1]  # get the index of the max log-probability
                x = pred.view(-1, fixed_size[0], fixed_size[1], fixed_size[2], 1).float()
                scipy.misc.imsave('Images/Test/p-{}-{}-3.jpg'.format(i, epoch),
                                  x[0, :, :, fixed_size[2] // 2, 0].cpu().numpy().T)
                """
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

    model = ResVNet.ResVNet4()
#    model = DenseVNet.DenseNet_FullVNet()

    print('Using', torch.cuda.device_count(), 'GPU(s).')
    model = nn.DataParallel(model)
    model = model.to(device)

    model.apply(AtriaSeg2018.weights_init)

    check_dir(output_dir)

    fwriter_train = open(output_dir + '/train-mode{}-fold{}.csv'.format(train_mode, fold_K), 'w')
    fwriter_test = open(output_dir + '/test-mode{}-fold{}.csv'.format(train_mode, fold_K), 'w')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
#    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200])
#    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=1e-8)
#    optimizer = optim.RMSprop(model.parameters(), weight_decay=1e-8)

    #    print('Preparing training data...')
    train_dataset = AtriaSeg2018.Dataset(train_paths, train_mode, transform=is_use_daug,
                                         normalization=is_use_normalization)
    train_loader = tdata.DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                    worker_init_fn=_init_fn)

#    print('Preparing testing data...')
    test_dataset = AtriaSeg2018.Dataset(test_paths, train_mode, transform=False,
                                        normalization=is_use_normalization)
    test_loader = tdata.DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False,
                                   worker_init_fn=_init_fn)

    target_mean = 0.06
    bg_weight = target_mean / (1. + target_mean)
    fg_weight = 1. - bg_weight
    class_weights = torch.FloatTensor([bg_weight, fg_weight])
    class_weights = class_weights.to(device)

    train_time, test_time = 0, 0

    max_dice = 0

    for epoch in range(epochs):
        start = tic()
        train(epoch, model, train_loader, optimizer, class_weights, fwriter_train)
        train_time += toc(start)
        start = tic()
        dice = test(epoch, model, test_loader, class_weights, fwriter_test)
        if dice > max_dice:
            # Save the model checkpoint
            torch.save(model.state_dict(), output_dir + '/vnet-mode{}-fold{}.ckpt'.format(train_mode, fold_K))
            max_dice = dice
        test_time += toc(start)
#        scheduler.step()
        print('Train time accumulated: {:.2f}s, Test time accumulated: {:.2f}s\n'.format(train_time, test_time))
    # Save the model checkpoint
    torch.save(model.state_dict(), output_dir + '/vnet-mode{}-fold{}-final.ckpt'.format(train_mode, fold_K))

    print('-------Max dice: {}--------'.format(max_dice))


if __name__ == '__main__':
    main()
