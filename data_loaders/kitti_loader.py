import torchvision.transforms as transforms
from PIL import Image  # use pillow-simd for speed, read https://docs.fast.ai/performance.html
import numpy as np
import os

from .mono_loader import MonoLoader
from .transforms import DoTest
from utils.kitti_utils import get_focal_length_baseline


class KittiLoader(MonoLoader):
    """
        Base class for loading kitti data, loading in the color images.
        Extended in the KittiRawDepthLoader & KittiDepthLoader.
    """

    def __init__(self,  *args, **kwargs):
        super(KittiLoader, self).__init__(*args, **kwargs)

        # Get a transform for these images.
        transform_functions = [transforms.ToTensor()]
        if self.mode == 'test':
            transform_functions.append(DoTest())
        self.transform = transforms.Compose(transform_functions)

        self.resize_transform = transforms.Resize(self.size)

    def get_image_path(self, path_tuple, load_left_image):
        # If we load the left image, we need the image_02 folder, else image_03
        image_folder = 'image_0{}/data/'.format(2 if load_left_image else 3)
        # Concat all pieces of the path and return.
        return os.path.join(path_tuple[0], image_folder, path_tuple[1] + self.extension)

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

    def get_depth(self, path_tuple, do_flip):
        pass

    def get_camera_parameters(self, path_tuple):
        calib_dir = '/'.join(path_tuple[0].split('/')[:-1])
        return get_focal_length_baseline(calib_dir)


class KittiDepthLoader(KittiLoader):
    """
        KITTI dataset which uses the updated ground truth depth maps
    """

    def __init__(self, *args, **kwargs):
        super(KittiDepthLoader, self).__init__(*args, **kwargs)

    def get_depth_path(self, path_tuple):
        path_string = "proj_depth/groundtruth/image_02/{}{}".format(path_tuple[1], self.extension)
        return os.path.join(path_tuple[0], path_string)

    def get_depth(self, path_tuple, do_flip):
        depth_path = self.get_depth_path(path_tuple)

        depth_gt = Image.open(depth_path)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
