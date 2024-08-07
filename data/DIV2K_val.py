import torch.utils.data as data
from os.path import join, exists
from os import listdir
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import numpy as np


def img_modcrop(image, modulo):
    sz = image.size
    w = np.int32(sz[0] / modulo) * modulo
    h = np.int32(sz[1] / modulo) * modulo
    out = image.crop((0, 0, w, h))
    return out


def np2tensor():
    return Compose([
        ToTensor(),
    ])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])


def load_image(filepath):
    return Image.open(filepath).convert('RGB')


class DIV2KValidSet(data.Dataset):
    def __init__(self, hr_dir, lr_dir, upscale):
        super(DIV2KValidSet, self).__init__()

        if not exists(hr_dir):
            raise ValueError(f"High-resolution directory {hr_dir} does not exist.")
        if not exists(lr_dir):
            raise ValueError(f"Low-resolution directory {lr_dir} does not exist.")

        self.hr_filenames = sorted([join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)])
        self.lr_filenames = sorted([join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)])
        self.upscale = upscale

        if not self.hr_filenames:
            raise ValueError(f"No valid high-resolution images found in {hr_dir}.")
        if not self.lr_filenames:
            raise ValueError(f"No valid low-resolution images found in {lr_dir}.")

    def __getitem__(self, index):
        input = load_image(self.lr_filenames[index])
        target = load_image(self.hr_filenames[index])
        input = np2tensor()(input)
        target = np2tensor()(img_modcrop(target, self.upscale))

        return input, target

    def __len__(self):
        return len(self.lr_filenames)

# Example usage for testing the existence of directories and files:
if __name__ == "__main__":
    hr_dir = "path_to_high_resolution_images"
    lr_dir = "path_to_low_resolution_images"
    upscale = 2
    try:
        dataset = DIV2KValidSet(hr_dir, lr_dir, upscale)
        print(f"Dataset loaded with {len(dataset)} images.")
    except ValueError as e:
        print(e)
