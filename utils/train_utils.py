import numpy as np
import os

from config import EIGEN_PATH, CITYSCAPES_PATH


def check_if_all_images_are_present(dataset, data_dir_path):
    set_names = ['train', 'val', 'test']

    if dataset == 'kitti':
        data_set_path = EIGEN_PATH
    elif dataset == 'cityscapes':
        data_set_path = CITYSCAPES_PATH
    else:
        raise ValueError("Code for running that dataset has not yet been implemented")

    all_file_names_paths = []
    for set_name in set_names:
        full_path_to_set = data_set_path.format(set_name)
        with open(full_path_to_set, 'r') as f:
            data_set_names = f.read()
            data_set_names = data_set_names.splitlines()
        for names in data_set_names:
            names = names.split(' ')
            all_file_names_paths.extend(names)

    total_length = len(all_file_names_paths)
    number_not_present = 0
    for file_path in all_file_names_paths:
        full_file_path = os.path.join(data_dir_path, file_path)
        if os.path.exists(full_file_path):
            number_not_present += 1

    print("-- Dir check for {}: --".format(dataset))
    print("{} out of {} images present".format(number_not_present, total_length))
    return


def get_present_images_from_list(lst_of_paths, data_dir):
    to_remove = []
    for pos, line in enumerate(lst_of_paths):
        line = line.split()
        single_left_path = os.path.join(data_dir, line[0])
        if not os.path.exists(single_left_path):
            to_remove.append(pos)

    for pos in to_remove[::-1]:
        lst_of_paths.pop(pos)
    return lst_of_paths


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)

    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return (r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp).astype(np.float32)


def print_epoch_update(epoch, time, losses):
    # Print update for this epoch.
    train_loss = losses[epoch]['train']
    val_loss = losses[epoch]['val']

    update_print_statement = 'Epoch: {}\t | train: {:.4f}\t | val: {:.4f}\t | time: {:.2f}'
    print(update_print_statement.format(epoch, train_loss, val_loss, time))

    # Print loss weights after the epoch, if it is applicable.
    if 'alphas' in losses[epoch].keys():
        weight_str = 'Loss Weights: '
        for weight in losses[epoch]['alphas']:
            weight_str += '{:.4f} '.format(weight)
        print(weight_str)
    return


def pre_validation_update(val_loss):
    val_loss_string = 'Pre-training val loss:\t{:.4f}'.format(val_loss)
    print(val_loss_string)
    return
