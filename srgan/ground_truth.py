import os
from PIL import Image

class GroundTruthGenerator:
    def __init__(self, image_folder):

        """ image_folder: the path to the folder containing the images to be processed"""
        self.image_folder = image_folder
        # get the list of image files in the folder
        self.image_paths = [
            os.path.join(image_folder, fname)
            for fname in os.listdir(image_folder)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
        ]

        if not self.image_paths:
            raise ValueError("no images found in the specified folder")

    def generate_ground_truth(self, output_folder):
        """
        output_folder: directory where the resized images will be saved
       
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for img_path in self.image_paths:
            # open the image file
            input_image = Image.open(img_path)
            # resize the image to 256x256 pixels
            resized_image = input_image.resize((256, 256), Image.BILINEAR)
            
            # construct the path to save the resized image
            img_name = os.path.basename(img_path)  # get the image file name
            output_path = os.path.join(output_folder, img_name)
            
            # save picture
            resized_image.save(output_path)
            print(f"already process: {output_path}")

# test
if __name__ == "__main__":
    # input and output folders
    input_folder = './test_images2'
    output_folder = './resized_images'

    # create an instance of the GroundTruthGenerator class
    gen = GroundTruthGenerator(input_folder)
    gen.generate_ground_truth(output_folder)
