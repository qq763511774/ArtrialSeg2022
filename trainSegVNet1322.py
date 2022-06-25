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

# number of processors for data processing
num_workers = 8

# training epochs
epochs = 300

# mini-batch size, net1->4, net2->1, as described in the paper
batch_size = 1

# random seed
seed = 0

# input data dir
data_dir = 'Training Set 30'

# output dir
output_dir = 'Out'

# use data augmentation
is_use_daug = True

# intensity normalization & histogram equalization for input
is_use_normalization = True

is_output_train_images = False
is_output_test_images = False

# --------------------------------

# get all full paths of folders under data_dir
All_paths = AtriaSeg2018.get_all_paths(data_dir)

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

        loss = AtriaSeg2018.dice_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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


def main():

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = ResVNet.ResVNet8()
#    model = DenseVNet.DenseNet_FullVNet()

    print('Using', torch.cuda.device_count(), 'GPU(s).')
    model = nn.DataParallel(model)
    model = model.to(device)

    model.apply(AtriaSeg2018.weights_init)

    check_dir(output_dir)

    fwriter_train = open(output_dir + '/train.csv', 'w')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
#    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200])
#    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=1e-8)
#    optimizer = optim.RMSprop(model.parameters(), weight_decay=1e-8)

    #    print('Preparing training data...')
    train_dataset = AtriaSeg2018.Dataset13(All_paths)
    train_loader = tdata.DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                    worker_init_fn=_init_fn)

    train_time = 0

    for epoch in range(epochs):
        start = tic()
        train(epoch, model, train_loader, optimizer, fwriter_train)
        train_time += toc(start)
        if epoch % 20 == 0:
            # Save the model checkpoint
            torch.save(model.state_dict(), output_dir + '/vnet-pretrained.ckpt')
#        scheduler.step()
        print('Train time accumulated: {:.2f}s.\n'.format(train_time))
    # Save the model checkpoint
    torch.save(model.state_dict(), output_dir + '/vnet-pretrained.ckpt')


if __name__ == '__main__':
    main()
