from PIL import Image, ImageFilter, ImageOps
import os

# Specify the folder containing images
image_folder = "datasets/img_train"  # Folder containing images
output_folder = "datasets/img_train2"  # Folder to save blurred images

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of all .jpg files in the folder
jpg_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

# Dictionary to store the blurred images
blurred_images = {}

# Iterate over each file, apply the blur, and save
for file in jpg_files:
    file_path = os.path.join(image_folder, file)
    try:
        # Open the image using Pillow
        with Image.open(file_path) as image:
            # Automatically adjust image orientation based on EXIF data
            image = ImageOps.exif_transpose(image)
            
            # Apply Gaussian blur filter
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=20))  # Adjust radius as needed

            # Save the blurred image to the output folder
            output_path = os.path.join(output_folder, file)
            blurred_image.save(output_path)

            # Optionally, store the blurred image in the dictionary
            blurred_images[file] = blurred_image

            print(f"Blurred and saved image: {file}")
    except Exception as e:
        print(f"Error processing image {file}: {e}")

# Print summary
print(f"Total number of blurred images saved: {len(blurred_images)}")
