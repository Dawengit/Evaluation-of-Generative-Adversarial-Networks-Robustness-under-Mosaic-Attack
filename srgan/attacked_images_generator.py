from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import os

import os
from PIL import Image, ImageOps
import numpy as np
import cv2

# Mosaic attack intensity parameter (block size), the larger the value, the more obvious the mosaic
# mosaic_level = 30
mosaic_level = 70

input_folders = ['./datasets/img_train2']
output_folder = 'images_dataset_attacked'

os.makedirs(output_folder, exist_ok=True)

output_counter = 0

for input_folder in input_folders:
    # Get all .jpg files in the folder
    jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    
    # Sort the files based on the numerical value in the filename
    jpg_files_sorted = sorted(jpg_files, key=lambda x: int(os.path.splitext(x)[0]))
    
    for img_filename in jpg_files_sorted:
        img_path = os.path.join(input_folder, img_filename)
        
        if not os.path.isfile(img_path):
            print(f'Cannot find image {img_path}')
            continue
        
        try:
            # Open the original image (PIL)
            with Image.open(img_path) as img_pil:
                # Automatically adjust image orientation based on EXIF data
                img_pil = ImageOps.exif_transpose(img_pil)

                # Convert the original image to a numpy array (RGB)
                img_np = np.array(img_pil.convert('RGB'))
                
                # Apply mosaic effect to the image (using OpenCV)
                height, width = img_np.shape[:2]
                blur = max(1, mosaic_level)

                # Resize down and then up to create the mosaic effect
                small_img = cv2.resize(img_np, (width // blur, height // blur), interpolation=cv2.INTER_LINEAR)
                mosaic_img = cv2.resize(small_img, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # Convert mosaic_img back to a PIL image
                mosaic_pil = Image.fromarray(mosaic_img)

                # Define the output path with a filename distinguished by a counter
                output_filename = f'attacked_{output_counter}.jpg'
                output_path = os.path.join(output_folder, output_filename)
                
                # Save the image
                mosaic_pil.save(output_path)
                print(f'Saved {output_path}')
                
                # Increment the counter
                output_counter += 1

        except Exception as e:
            print(f"Error processing image {img_filename}: {e}")

print(f"Total of {output_counter} processed mosaic images saved.")
