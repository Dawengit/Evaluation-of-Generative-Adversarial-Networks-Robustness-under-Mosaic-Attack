from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import os

# Adjustable blur radius (Gaussian blur)
blur_radius = 20

# Mosaic attack parameter (block size). The larger the value, the more obvious the mosaic
mosaic_level = 70

input_folders = ['archive/train/men', 'archive/train/women']
output_folder = 'images_dataset_attacked'

os.makedirs(output_folder, exist_ok=True)

output_counter = 0

for input_folder in input_folders:
    # Get all .jpg files in the folder
    jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    
    # Sort by the numeric part of the filename
    jpg_files_sorted = sorted(jpg_files, key=lambda x: int(os.path.splitext(x)[0]))
    
    for img_filename in jpg_files_sorted:
        img_path = os.path.join(input_folder, img_filename)
        
        if not os.path.isfile(img_path):
            print(f'Unable to find image {img_path}')
            continue
        
        try:
            # Open the original image (PIL)
            with Image.open(img_path) as img_pil:
                # Automatically adjust image orientation based on EXIF data
                img_pil = ImageOps.exif_transpose(img_pil)
                
                # Apply Gaussian blur
                blurred_img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                
                # Convert both the original clear image and the blurred image to numpy arrays (RGB format)
                img_np = np.array(img_pil.convert('RGB'))
                blurred_np = np.array(blurred_img_pil.convert('RGB'))
                
                # Apply mosaic to the blurred image (using cv2)
                # Get dimensions
                height, width = blurred_np.shape[:2]

                blur = max(1, mosaic_level)
                
                # Resize down and then back up
                small_img = cv2.resize(blurred_np, (width // blur, height // blur), interpolation=cv2.INTER_LINEAR)
                mosaic_img = cv2.resize(small_img, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # Final output concatenation: horizontally combine the mosaic image (attacked image) and the original clear image
                combined_np = np.hstack((mosaic_img, img_np))
                
                # Convert back to PIL image to save
                combined_img = Image.fromarray(combined_np)
                
                # Define output path, add "attacked" to the filename
                output_filename = f'combined_attacked_{output_counter}.jpg'
                output_path = os.path.join(output_folder, output_filename)
                
                # Save the image
                combined_img.save(output_path)
                print(f'Saved {output_path}')
                
                # Increment counter
                output_counter += 1

        except Exception as e:
            print(f"Error processing image {img_filename}: {e}")

print(f"A total of {output_counter} attacked (mosaic) combined images have been saved.")
