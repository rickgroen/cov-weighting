import os
from pathlib import Path


class RestrictedFilePathCreator:
    """
        Handles the case where we want to train on a subset of data.
    """

    def __init__(self, ratio, all_file_paths, data_dir_path, dataset):
        self.ratio = ratio
        self.all_files_paths = all_file_paths
        self.data_dir_path = data_dir_path
        self.dataset = dataset

        if ratio == 0.0:
            self.__create_file_names_from_existing_data()
        elif 0.0 < self.ratio < 1.0:
            self.__create_split_from_file_names()
        else:
            raise ValueError("Class should only be called when 0.0 <= ratio < 1.0")

    def __create_file_names_from_existing_data(self):
        file_extentsion_in_data_dir = self.__get_extention()

        to_remove = []
        for pos, line in enumerate(self.all_files_paths):
            if self.dataset == 'kitti':
                path_string = os.path.join(line[0], 'image_02', 'data', line[1] + file_extentsion_in_data_dir)
            else:
                path_string = os.path.join(line[0].format('leftImg8bit'),
                                           line[1].format('leftImg8bit') + file_extentsion_in_data_dir)

            if not os.path.exists(path_string):
                to_remove.append(pos)

        for pos in to_remove[::-1]:
            self.all_files_paths.pop(pos)

    def __create_split_from_file_names(self):
        split_at = int(self.ratio * len(self.all_files_paths))
        self.all_files_paths = self.all_files_paths[:split_at]

    def __get_extention(self):
        all_files = list(Path(self.data_dir_path).rglob("*.[pj][np][g]"))

        # Get only the paths from the folders of interest, depending on the dataset.
        if self.dataset == 'kitti':
            folders_with_data = ['training', 'testing', 'disparity']
        else:
            folders_with_data = ['leftImg8bit', 'rightImg8bit', 'disparity']

        exts_set = {os.path.splitext(str(name))[1] for name in all_files
                    if not any(substring in str(name) for substring in folders_with_data)}
        if len(exts_set) == 1:
            return list(exts_set)[0]
        else:
            raise FileExistsError('There is a mix of jps and pngs in your data dir. Revert to 1 kind.')
