import numpy as np
import SimpleITK as sitk
import pandas as pd
import os
import csv

data_dir = '/data/AtriaSeg2018/Merged'
#data_dir = 'Training Set 100'

mask_file_name = 'mask.nrrd'

csv_file_name = 'submission_QX-BUAA.csv'

sample_csv_name = 'submission.csv'

folders = []

with open(sample_csv_name, 'r') as sample_csv_reader:
    reader = csv.DictReader(sample_csv_reader)
    num = 0
    for row in reader:
        if num == 0:
            folders.append(row['ImageId'].split('_')[0])
        num += 1
        if num == 88:
            num = 0

print(folders)


def get_all_dirs(root):
    dirs = os.listdir(root)
    dirs.sort()
    return dirs


# this function loads .nrrd files into a 3D matrix and outputs it
# 	the input is the specified file path to the .nrrd file
def load_nrrd(full_path_filename):
    data = sitk.ReadImage(full_path_filename)
    data = sitk.Cast(sitk.RescaleIntensity(data), sitk.sitkUInt8)
    data = sitk.GetArrayFromImage(data)

    return (data)


# this function encodes a 2D file into run-length-encoding format (RLE)
# 	the inpuy is a 2D binary image (1 = positive), the output is a string of the RLE
def run_length_encoding(input_mask):
    dots = np.where(input_mask.T.flatten() == 1)[0]

    run_lengths, prev = [], -2

    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1
        prev = b

    return (" ".join([str(i) for i in run_lengths]))


#dirs = get_all_dirs(data_dir)
dirs = folders

is_first_write = True

for dir in dirs:

    print(dir)

    mask = load_nrrd(data_dir + '/' + dir + '/' + mask_file_name)
    print(mask.shape)
    mask[mask.nonzero()] = 1

    # encode in RLE
    image_ids = [dir + '_Slice_' + str(i) for i in range(mask.shape[0])]

    encode_cavity = []
    for i in range(mask.shape[0]):
        encode_cavity.append(run_length_encoding(mask[i, :, :]))

    # output to csv file
    csv_output = pd.DataFrame(data={"ImageId": image_ids, 'EncodeCavity': encode_cavity},
                              columns=['ImageId', 'EncodeCavity'])
    if is_first_write:
        csv_output.to_csv(csv_file_name, sep=",", index=False, mode='a', header=True)
        is_first_write = False
    else:
        csv_output.to_csv(csv_file_name, sep=",", index=False, mode='a', header=False)
