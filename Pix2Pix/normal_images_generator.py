from PIL import Image, ImageFilter, ImageOps
import numpy as np
import os
import matplotlib.pyplot as plt

# Adjustable blur radius parameter (for Gaussian blur). 
# The larger the value, the stronger the blur.
blur_radius = 20  # You can modify this value to adjust the blur intensity.

# Lists of input and output directories
input_folders = ['archive/train/men', 'archive/train/women']
output_folder = 'images_dataset'

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Flag for showing the example demonstration only once
example_shown = False

# Initialize output file counter
output_counter = 0

# Iterate over each input folder
for input_folder in input_folders:
    # Get all .jpg files in the folder
    jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    
    # Sort the files by the numeric part of the filename to ensure correct processing order
    # Assuming filenames are like '0.jpg', '1.jpg', ..., '108.jpg'
    jpg_files_sorted = sorted(jpg_files, key=lambda x: int(os.path.splitext(x)[0]))
    
    # Iterate over the sorted file list
    for img_filename in jpg_files_sorted:
        img_path = os.path.join(input_folder, img_filename)
        
        # Check if the image file exists
        if not os.path.isfile(img_path):
            print(f'Unable to find image {img_path}')
            continue
        
        try:
            # Open the original image
            with Image.open(img_path) as img:
                # Automatically adjust image orientation based on EXIF data
                img = ImageOps.exif_transpose(img)
                
                # Apply Gaussian blur
                blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                
                # Ensure both images have the same mode before concatenation
                if img.mode != blurred_img.mode:
                    blurred_img = blurred_img.convert(img.mode)
                
                # Ensure both images have the same size
                if img.size != blurred_img.size:
                    blurred_img = blurred_img.resize(img.size)
                
                # Convert PIL images to NumPy arrays
                img_np = np.array(img)
                blurred_np = np.array(blurred_img)
                
                # Check if the image is grayscale or has alpha channel, and adjust shapes accordingly
                if img_np.ndim == 2:  # Grayscale
                    img_np = np.expand_dims(img_np, axis=2)
                    blurred_np = np.expand_dims(blurred_np, axis=2)
                elif img_np.ndim == 3 and img_np.shape[2] == 4:  # RGBA, need to convert to RGB
                    img_np = img_np[:, :, :3]
                    blurred_np = blurred_np[:, :, :3]
                
                # Ensure both images have the same number of channels
                if img_np.shape[2] != blurred_np.shape[2]:
                    print(f'Channel mismatch, skipping {img_filename}')
                    continue
                
                # Concatenate the images horizontally
                combined_np = np.hstack((blurred_np, img_np))
                
                # Convert the concatenated array back to a PIL image
                combined_img = Image.fromarray(combined_np)
                
                # Define the output path, filename as combined_{output_counter}.jpg
                output_filename = f'combined_{output_counter}.jpg'
                output_path = os.path.join(output_folder, output_filename)
                
                # Save the concatenated image
                combined_img.save(output_path)
                print(f'Saved {output_path}')
                
                # Show an example comparison of before and after blur (only once)
                if not example_shown:
                    plt.figure(figsize=(10, 5))

                    # Show the original image
                    plt.subplot(1, 2, 1)
                    plt.imshow(img)
                    plt.title('Original Image')
                    plt.axis('off')

                    # Show the blurred image
                    plt.subplot(1, 2, 2)
                    plt.imshow(blurred_img)
                    plt.title('Blurred Image')
                    plt.axis('off')

                    plt.show()
                    example_shown = True  # Only show the example once
                
                # Increment the output file counter
                output_counter += 1
        
        except Exception as e:
            print(f"Error processing image {img_filename}: {e}")

# Print a summary
print(f"A total of {output_counter} blurred and concatenated images were saved.")
