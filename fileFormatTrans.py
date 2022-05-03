#不需要
import os
import SimpleITK as sitk
import numpy as np
from skimage import img_as_ubyte

data_dir = 'Testing Set 54'

folders = os.listdir(data_dir)
folders.sort()

data_dir_HNet = '/data/AtriaSeg2018/HNet_prob'

def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def check_dir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)


for fold in folders:
    print(fold)
    path = data_dir_HNet + '/' + fold + '.npz'
    p = np.load(path)
    p = p['arr_0']
#    p = softmax(p, axis=0)
    p0 = p[0, :, :, :]
    p0 = (p0 - np.min(p0)) / (np.max(p0) - np.min(p0))
    p1 = p[1, :, :, :]
    p1 = np.nan_to_num(p1)
    print(np.max(p1))
    print(np.min(p1))
    p1 = (p1 - np.min(p1)) / (np.max(p1) - np.min(p1))

    check_dir(data_dir_HNet + '/' + fold)

    image = img_as_ubyte(p0)
    sitk.WriteImage(sitk.GetImageFromArray(image), data_dir_HNet + '/' + fold + '/p0.nrrd')
    image = img_as_ubyte(p1)
    sitk.WriteImage(sitk.GetImageFromArray(image), data_dir_HNet + '/' + fold + '/p1.nrrd')