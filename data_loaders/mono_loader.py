import random
import os
from torch.utils.data import Dataset
from .transforms import AugmentImage

from utils.reduce_image_set import RestrictedFilePathCreator
from config import IMAGE_EXTENSION


class MonoLoader(Dataset):
    def __init__(self, data_dir, path_to_file_paths, mode, train_ratio, size, augment_parameters):
        super(MonoLoader, self).__init__()

        with open(path_to_file_paths, 'r') as f:
            all_paths = f.read().splitlines()

        all_split_paths = [tuple([os.path.join(data_dir, pair.split(' ')[0]),
                                  pair.split(' ')[1]]) for pair in all_paths]

        dataset = 'kitti' if 'kitti' in str(self.__class__).lower() else 'cityscapes'
        # If we want a subset of data, now is the time to select it.
        if train_ratio != 1.0:
            restricted_creator = RestrictedFilePathCreator(train_ratio, all_split_paths, data_dir, dataset)
            all_split_paths = restricted_creator.all_files_paths

        self.paths = all_split_paths

        self.transform = None
        self.size = size
        self.augment_parameters = augment_parameters
        self.augment_transform = AugmentImage(self.augment_parameters)

        self.mode = mode
        # Setting this to png, but feel free to change in config_parameters, if you had saved as jpg.
        self.extension = IMAGE_EXTENSION

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.get_train_sample(idx)
        elif self.mode == 'val':
            return self.get_val_sample(idx)
        return self.get_test_sample(idx)

    def get_train_sample(self, idx):
        # Check if we should perform augmentation.
        do_color_aug = random.random() > 0.5
        do_flip = random.random() > 0.5
        # Set the random parameters of the colour augmentation class.
        if do_color_aug:
            self.augment_transform.set_random_variables()
        # Retrieve images, note that flipping also means changing the left and right images.
        left_image = self.get_image(self.paths[idx], do_color_aug, do_flip, left=not do_flip)
        right_image = self.get_image(self.paths[idx], do_color_aug, do_flip, left=do_flip)
        return {'left_image': left_image, 'right_image': right_image}

    def get_val_sample(self, idx):
        left_image = self.get_image(self.paths[idx], False, False, left=True)
        right_image = self.get_image(self.paths[idx], False, False, left=False)
        return {'left_image': left_image, 'right_image': right_image}

    def get_test_sample(self, idx):
        left_image = self.get_image(self.paths[idx], False, False, left=True)
        # On test, retrieve the ground truth depth. Also flip the left image. Additionally, retrieve
        # baseline and focal length
        gt_depth = self.get_depth(self.paths[idx], False)
        focal_length, baseline = self.get_camera_parameters(self.paths[idx])
        return {'left_image': left_image, 'gt_depth': gt_depth, 'camera': (focal_length, baseline)}

    def get_image(self, path_tuple, do_color_aug, do_flip, left=True):
        pass

    def get_depth(self, path_tuple, do_flip):
        pass

    def get_camera_parameters(self, path_tuple):
        pass
