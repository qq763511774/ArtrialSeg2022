from skimage.feature import blob_dog
import AtriaSeg2022
import DATA_INFO as DI

# for i in range(DI.TRAIN_DATA_NUM):
#     if i % 5 == 0:
#         volume_data = AtriaSeg2022.load_niigz(DI.TRAIN_DATA_DIR+'/train_'+str(i+1)+'/'+DI.TRAIN_VOLUME_NAME)
#         blobs_dog = blob_dog(volume_data)
#         print(str(i+1)+'\n'+blob_dog.shape)

