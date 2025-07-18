import os
import random
from typing import Callable, Sequence

import numpy as np
import torch.utils.data
from PIL import Image

from .utils import get_class_counts


class CuneiformClassification(torch.utils.data.Dataset):
    """
    Classification dataset for to be used in training and evaluating of cuneiform classification tasks. This method
    expects that sign crops are made and available in root/sign_crops.
    """
    def __init__(self, root: str, split: str = 'train', min_nb_img: int = 20, transform: Callable = None, lazy_load = True,
                 img_type: Sequence[str] = None, tablet_type: str = None, split_file = 'train_val_split_08_02.txt'):
        """
        :param root: the root directory where both annotations and sign crops are stored
        :param split: either 'train' or 'val' to load training or validation samples.
        :param min_nb_img: the minimal number of images a class needs to have to be included in the dataset.
        :param transform: image transformation to be applied before returning a sample
        :param lazy_load: if True, images are not loaded on creation on self.images contains file paths. If False,
        images are loaded into memory and self.images contains the raw data samples.
        :param img_type: visualization types that are included in this dataset (e.g., ColorA, SketchB...)
        :param tablet_type: which tablet types to include (HS, CUNES...). Only works if these substrings also appear
        in the file names of the sign crops, which should be the case if they are created correctly.
        :param split_file: the split file to use. This should list all the sign crops to include, but 'type' agnostic.
        Instead of having e.g., SketchB, it should include the file paths with the generic 'TYPE'. This file should not
        be manually created and be included from the creation process.
        """
        self.split_file = os.path.join(root, split_file)
        self.sign_root = os.path.join(root, 'sign_crops')
        self.split = split

        self.min_nb_img = min_nb_img
        self.transform = transform
        self.img_type = img_type if isinstance(img_type, list) else [img_type]
        self.tablet_type = tablet_type if isinstance(tablet_type, list) else [tablet_type]

        self.class_to_idx = self.create_cls_to_idx()
        self.idx_to_class = {self.class_to_idx[cls]: cls for cls in self.class_to_idx}
        self.class_counts = get_class_counts(f"{self.sign_root}/../")

        # TODO: if not lazy loading than should add a list (?) with image names so that it is recoverable
        self.lazy_load = lazy_load
        self.images, self.labels = self.load_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        if self.lazy_load:
            img = Image.open(os.path.join(self.sign_root, self.images[item]))
        else:
            img = self.images[item]

        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[item]

        return img, label


    def create_cls_to_idx(self):
        """
        Creates a class to index list based on the folders that are inside sign_root. They are alphabetically ordered,
        and the categories that have less than self.min_nb_img are skipped. This means that if the number of examples
        changes, or a new category with more than self.min_nb_img is added, the output_idx of a model will be different.
        As the indexes of the output need to be dense and consecutive, there is no easy way around this issue.

        We cannot directly use the number of files in a sign category folder as there may be multiple visualizations
        of the same sign. It uses sign_map.csv, which is created during the sign crop process.
        :return: cls_to_idx dict with sign_names to idx.
        """
        class_counts = get_class_counts(f"{self.sign_root}/../")

        all_classes = sorted([f for f in os.listdir(self.sign_root) if os.path.isdir(os.path.join(self.sign_root, f))])
        cls_to_idx = {}
        counter = 0
        for cls in all_classes:
            if class_counts[cls] >= self.min_nb_img:
                cls_to_idx[cls] = counter
                counter += 1
        return cls_to_idx

    def validate_img(self, label: str, file_path: str):
        """
        Return true if image should be in the dataset, return false otherwise
        :return:
        """
        if self.class_counts[label] < self.min_nb_img:
            return False

        # If no tablet type is specified, include sample.
        if self.tablet_type[0] is None:
            return True
        # If tablet types are specified, check whether they appear as substring in the file name.
        # This is not safe, but works for our purposes (splitting HS/CUNES/VAT/O).
        for t_type in self.tablet_type:
            if t_type in file_path:
                return True

        return False

    def load_images(self):
        """
        Loads the images and labels into the dataset. It does so by reading the split_file, which lists all the sign
        crops that were made during the dataset creation and whether it belongs to the train or validation data.
        :return:
        """
        all_images, all_labels = [], []

        with open(os.path.join(self.split_file)) as f:

            for line in f.readlines():
                # [sign_map/file_name, train (0/1), val (0/1), test (0/1)
                sample_info = line.strip().split(' ')
                # The first part of sign_name is the label
                cls_label = sample_info[0].split('/')[0]

                if not self.validate_img(cls_label, sample_info[0]):
                    continue

                for img_type in self.img_type:
                    sample_name = sample_info[0].replace('TYPE', img_type)
                    if (self.split == 'train' and sample_info[1] == '1') or \
                        (self.split == 'val' and sample_info[2] == '1'):
                        if self.lazy_load:
                            if os.path.isfile(os.path.join(self.sign_root, sample_name)):
                                all_images.append(sample_name)
                                all_labels.append(self.class_to_idx[cls_label])
                            else:
                                continue
                                # print(f"Image not found: {sample_name}")
                        else:
                            try:
                                img = Image.open(os.path.join(self.sign_root, sample_name))
                                all_images.append(img.copy())
                                all_labels.append(self.class_to_idx[cls_label])
                                img.close()
                            except FileNotFoundError:
                                continue
                                # print(sample_name)
        if self.lazy_load:
            # Allows lazy load images to be used as index, but with full img arrays this can't be done because their
            # size is different.
            all_images = np.array(all_images)
        return all_images, np.array(all_labels)


