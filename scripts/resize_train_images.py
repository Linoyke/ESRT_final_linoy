import os
import argparse
import numpy as np
import skimage.transform

def resize_image(image, scale):
    """Resize the image by the given scale factor and adjust pixel values."""
    height, width = image.shape[:2]
    new_height, new_width = int(height / scale), int(width / scale)
    
    # Resize the image
    resized_image = skimage.transform.resize(image, (new_height, new_width), anti_aliasing=True)
    
    # Rescale back to [0, 255] if the original image was in this range
    if resized_image.max() <= 1.0:
        resized_image = resized_image * 255.0

    return resized_image.astype(np.uint8)  # Convert to uint8 to match original image format

def process_images(input_dir, output_dir, scale):
    """Process all numpy arrays in the input directory, resize them, and save to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.npy'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)

                image = np.load(input_path)
                resized_image = resize_image(image, scale)
                np.save(output_path, resized_image)

                print(f'Resized and saved {file} to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize numpy array images')
    parser.add_argument('--input_dir', required=True, help='Path to the input directory containing numpy arrays')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory to save resized images')
    parser.add_argument('--scale', type=int, required=True, help='Scale factor to resize images')

    args = parser.parse_args()
    process_images(args.input_dir, args.output_dir, args.scale)
