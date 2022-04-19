import tarfile
from os import remove
from os.path import exists, join, basename

from six.moves import urllib
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from .dataset import DatasetFromFolder


"""
Div2k DataSet Reference
@InProceedings{Agustsson_2017_CVPR_Workshops,
	author = {Agustsson, Eirikur and Timofte, Radu},
	title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
	month = {July},
	year = {2017}
} 
Type: DataSet
Link: https://data.vision.ee.ethz.ch/cvl/DIV2K/
"""
def load_div2k(dest="./dataset"):
    output_image_dir = join(dest, "Div2k")

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    root_dir = load_div2k()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    root_dir = load_div2k()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
