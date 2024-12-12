

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # Import Pillow's Image module
# refence:  chatgpt 

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False, is_pred=False):
        data_type = "train" if not is_testing else "test"
        if is_pred:
            # Ensure 'test_images/' directory exists
            # test_dir = './sharp_data/0.jpg'
            # test_dir = './test_images2/'
            test_dir = './blurred_data/'
            #
            if not os.path.exists(test_dir):
                raise FileNotFoundError(f"The directory '{test_dir}' does not exist.")
            batch_images = [os.path.join(test_dir, x) for x in os.listdir(test_dir) if self.is_image_file(x)]
            if len(batch_images) == 0:
                raise ValueError(f"No images found in '{test_dir}'. Please add images to this directory.")
        else:
            path = glob(os.path.join(self.dataset_name, '*'))
            if len(path) == 0:
                raise ValueError(f"No images found in '{self.dataset_name}'. Please check the dataset path.")
            batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img = self.imread(img_path)

            # Resize images
            img_hr = self.imresize(img, self.img_res)
            img_lr = self.imresize(img, (self.img_res[0] // 4, self.img_res[1] // 4))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr).astype(np.float32) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr).astype(np.float32) / 127.5 - 1.

        return imgs_hr, imgs_lr
    
    # def generate_ground_truth(self, batch_size=1):

    def imread(self, path):
        """Reads an image from a path and converts it to RGB."""
        with Image.open(path) as img:
            return img.convert('RGB')

    def imresize(self, img, size):
        """Resizes an image to the specified size.

        Args:
            img (PIL.Image.Image or numpy.ndarray): Image to resize.
            size (tuple): Desired size as (width, height).

        Returns:
            numpy.ndarray: Resized image as a NumPy array.
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype('uint8'), 'RGB')
        resized_img = img.resize(size, Image.BICUBIC)
        return np.array(resized_img)

    def is_image_file(self, filename):
        """Checks if a file is an image based on its extension."""
        IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)
