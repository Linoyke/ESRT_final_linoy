import os
import argparse
import numpy as np
from skimage import io, transform
from skimage.util import img_as_ubyte

def resize_image(image, scale):
    """Resize the image by the given scale factor."""
    height, width = image.shape[:2]
    new_height, new_width = int(height / scale), int(width / scale)
    resized_image = transform.resize(image, (new_height, new_width), anti_aliasing=True)
    return resized_image

def process_images(input_dir, output_dir, scale):
    """Process all images in the input directory, resize them, and save to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)

                image = io.imread(input_path)
                resized_image = resize_image(image, scale)
                resized_image = img_as_ubyte(resized_image)  # Convert to uint8 before saving
                io.imsave(output_path, resized_image)

                print(f'Resized and saved {file} to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize image files')
    parser.add_argument('--input_dir', required=True, help='Path to the input directory containing image files')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory to save resized images')
    parser.add_argument('--scale', type=int, required=True, help='Scale factor to resize images')

    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir, args.scale)
