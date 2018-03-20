import numpy as np
import os
from os.path import join as pjoin
from scipy.misc import imresize
from imageio import imread
import time
import h5py

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

class Resnet:

    def __init__(self):
        self.model = ResNet50(include_top=False, weights='imagenet')
        

    def process(self, img_dir):        

        list_files = sorted(os.listdir(img_dir))

        firststep = True
        #Load picture and preprocess them
        files_path = []
        n_batch = 0
        n_pictures = 0

        for picture_file in list_files:
            if (os.path.isfile(pjoin(img_dir, picture_file)) and picture_file[-4:] == '.jpg'):

                # Preprocess the picture
                n_pictures += 1
                files_path.append(pjoin(img_dir, picture_file))
                temp = imread(pjoin(img_dir, picture_file))
                print(picture_file)
                print(temp.shape)
                temp = imresize(temp, (224,224)).astype("float32")
                temp = preprocess_input(temp[np.newaxis])

                # Append to a big numpy
                if firststep:
                    firststep = False
                    img_batch = temp
                else:
                    img_batch = np.vstack((img_batch, temp))

                # Compute representation and save it
                # if n_pictures == 400:
                #     n_pictures = 0
                #     n_batch += 1
                #     first_step = True
                #     print("Computing batch %s"%n_batch)
                #     out_tensor = self.model.predict(img_batch)
                #     out_tensor = out_tensor.reshape((-1, out_tensor.shape[-1]))
                #     print("output shape:", out_tensor.shape)

                #     # Serialize representations
                #     print("SAVING BATCH")
                #     h5f = h5py.File(
                #         pjoin(
                #             img_dir,
                #             "sorted_aug_img_emb_%s.h5"%n_batch
                #         ),
                #         'w'
                #     )
                #     h5f.create_dataset('img_emb', data=out_tensor)
                #     h5f.close()


        # Compute representations
        out_tensor = self.model.predict(img_batch, batch_size=32)
        out_tensor = out_tensor.reshape((-1, out_tensor.shape[-1]))
        print("output shape:", out_tensor.shape)

        # Serialize representations
        h5f = h5py.File(pjoin(img_dir, "sorted_aug_img_emb.h5"), 'w')
        h5f.create_dataset('img_emb', data=out_tensor)
        h5f.close()

        return out_tensor
