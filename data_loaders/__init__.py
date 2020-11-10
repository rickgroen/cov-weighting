from torch.utils.data import DataLoader

from config import EIGEN_PATH, CITYSCAPES_PATH
from .kitti_loader import KittiDepthLoader
from .cityscapes_loader import CityScapesLoader


def create_dataloader(args, explicit_mode):
    """ Yields a data loader given args and an explicit mode, such that we can do both
        training and validation splits without changing args.
    """
    # Get relevant parameters from args.
    data_dir = args.data_dir
    dataset = args.dataset
    train_ratio = args.train_ratio
    size = (args.input_height, args.input_width)
    augment_parameters = args.augment_parameters

    if dataset == 'kitti':
        files_path = EIGEN_PATH.format(explicit_mode)
        dataset = KittiDepthLoader(data_dir, files_path, explicit_mode, train_ratio, size, augment_parameters)
    elif dataset == 'cityscapes':
        files_path = CITYSCAPES_PATH.format(explicit_mode)
        dataset = CityScapesLoader(data_dir, files_path, explicit_mode, train_ratio, size, augment_parameters)
    else:
        raise ValueError("No other dataset has been implemented yet.")

    n_img = len(dataset)
    print('Loaded a dataset with {} images'.format(n_img))

    if explicit_mode == 'train':
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    # If val or test, then feed unshuffled raw data.
    return DataLoader(dataset, batch_size=1, shuffle=False,
                      num_workers=1, pin_memory=True)
