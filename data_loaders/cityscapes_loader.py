import torchvision.transforms as transforms
from PIL import Image  # use pillow-simd for speed, read https://docs.fast.ai/performance.html
import os
import torch
import numpy as np

from .mono_loader import MonoLoader
from .transforms import DoTest
from config import CITYSCAPES_FOCAL_LENGTH, CITYSCAPES_BASELINE


class CityScapesLoader(MonoLoader):

    """
        Base class for loading kitti data, loading in the color images.
        Extended in the KittiRawDepthLoader & KittiDepthLoader.
    """

    def __init__(self,  *args, **kwargs):
        super(CityScapesLoader, self).__init__(*args, **kwargs)

        # Get a transform for these images.
        transform_functions = [transforms.ToTensor()]
        if self.mode == 'test':
            transform_functions.append(DoTest())
        self.transform = transforms.Compose(transform_functions)

        self.resize_transform = transforms.Resize(self.size)

    def get_image_path(self, path_tuple, load_left_image):
        # If we load the left image, we need the leftImg8bit folder & identifier, else rightImg8bit
        identifier = 'leftImg8bit' if load_left_image else 'rightImg8bit'
        image_folder = path_tuple[0].format(identifier)
        file_name = path_tuple[1].format(identifier)
        # Concat all pieces of the path and return.
        return os.path.join(image_folder, file_name + self.extension)

    def get_image(self, path_tuple, do_color_aug, do_flip, left=True):
        # Open the image and apply transforms.
        image_path = self.get_image_path(path_tuple, left)
        image = Image.open(image_path)
        image = self.resize_transform(image)

        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = self.transform(image)

        # Do other augmentations, if necessary.
        if do_color_aug:
            image = self.augment_transform(image)

        return image

    def get_disparity_path(self, path_tuple):
        return os.path.join(path_tuple[0], path_tuple[1] + self.extension).format('disparity', 'disparity')

    def get_depth(self, path_tuple, do_flip):
        """ It is common to evaluate CityScapes purely on disparities, since the ground truth
            is available. Therefore this 'get_depth' function returns the ground truth disparities.
        """
        depth_path = self.get_disparity_path(path_tuple)
        disparity = Image.open(depth_path)
        # From plotting left, right images and disparity maps below each other,
        # I've confirmed the note in the documentation:
        # to obtain valid disparity values, apply:  p > 0: d = ( float(p) - 1. ) / 256
        # Where p is the disparity in pixels on 1024x2048 resolution.
        disparity = transforms.ToTensor()(np.array(disparity))
        # Apply:  p > 0: d = ( float(p) - 1. ) / 256
        disparity = disparity.type(torch.FloatTensor)
        disparity[torch.where(disparity > 0)] = (disparity[torch.where(disparity > 0)] - 1) / 256
        return disparity

    def get_camera_parameters(self, path_tuple):
        return CITYSCAPES_FOCAL_LENGTH, CITYSCAPES_BASELINE
