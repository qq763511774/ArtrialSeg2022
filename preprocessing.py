import SimpleITK as sitk
import AtriaSeg2022
from multiprocessing import Pool
def resampling(i):
    print('start{}'.format(i))
    volume_data = sitk.ReadImage(AtriaSeg2022.DI.TRAIN_DATA_DIR+'train_'+str(i+1)+'/enhanced.nii.gz')
    # volume_data = sitk.ReadImage(AtriaSeg2022.DI.TRAIN_DATA_DIR+'train_'+str(i+1)+'/atriumSegImgMO.nii.gz')

    # Use the original spacing (arbitrary decision).
    output_spacing = AtriaSeg2022.DI.NEW_SPACING
    # Identity cosine matrix (arbitrary decision).   
    # output_direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    output_direction = volume_data.GetDirection()
    # Minimal x,y coordinates are the new origin.
    output_origin = volume_data.GetOrigin()
    # Compute grid size based on the physical size and spacing.

    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(volume_data.GetSize(), volume_data.GetSpacing(), AtriaSeg2022.DI.NEW_SPACING)]
    resampled_image = sitk.Resample(volume_data, new_size, sitk.Transform(), sitk.sitkLinear, output_origin, output_spacing, output_direction)
    sitk.WriteImage(resampled_image,AtriaSeg2022.DI.TRAIN_DATA_DIR+'train_'+str(i+1)+'/enhanced_resampled.nii.gz')
    # sitk.WriteImage(resampled_image,AtriaSeg2022.DI.TRAIN_DATA_DIR+'train_'+str(i+1)+'/atriumSegImgMO_resampled.nii.gz')

if __name__ == '__main__':
    with Pool(8) as pool:
        pool.map(resampling,range(130))
