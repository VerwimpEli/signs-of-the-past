import json
import os
import random
import re
import time
from collections import Counter
from typing import List, Any

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.tv_tensors
from torch import dtype
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from PIL import Image
from matplotlib import pyplot as plt
from torchvision.datasets import CocoDetection

from .utils import *


class CuneiformDetection(CocoDetection):

    def __init__(self, img_root, ann_file, img_type='SketchB', *, min_nb_images = 20, single_label: bool = False, **kwargs):
        """
        :param single_label: if true, all category_ids are mapped to the same number (1) so that it becomes a pure
        recognition task.
        """
        super().__init__(img_root, ann_file, **kwargs)
        self.img_type = img_type
        self.single_label = single_label
        self.class_map, self.idx_to_name = self.create_class_map(min_nb_images)

    def _load_image(self, img_id: int) -> Image.Image:
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        path = path.replace('TYPE', self.img_type)
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, img_id: int) -> List[Any]:
        target = self.coco.loadAnns(self.coco.getAnnIds(img_id))

        for annot in target:
            if self.single_label:
                annot['category_id'] = 1
            else:
                # Adapt target if min_nb_images is higher than zero
                annot['category_id'] = self.class_map[annot['category_id']]
        return target

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img_id = self.ids[index]
        image = self._load_image(img_id)
        target = self._load_target(img_id)

        image = np.array(image, dtype=np.uint8)
        boxes = np.array([t['bbox'] for t in target])
        labels = np.array([t['category_id'] for t in target])

        if self.transforms is not None:
            # Albumentations
            augmented = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = augmented.pop('image')
            # Bbox conversion, PyTorch FasterRCNN requires Pascal Voc style (or perhaps only my own code, not sure).
            # TODO: this should probably also happen then if there are no augmentations?
            boxes = torchvision.tv_tensors.BoundingBoxes(augmented['bboxes'], format=BoundingBoxFormat.XYWH,
                                                         canvas_size=image.shape)
            boxes = torchvision.ops.box_convert(boxes, 'xywh', 'xyxy')
            target = {'boxes': boxes,
                      'labels': torch.tensor(augmented['labels'], dtype=torch.int64),
                      'image_id': target[0]['image_id']}

            # torchvision.v2
            # image, target = self.transforms(image, target)

        return image, target

    def load_image(self, img_id: int) -> Image.Image:
        return self._load_image(img_id)

    def load_target(self, img_id: int):
        return self._load_target(img_id)

    def get_image_path(self, img_id: int):
        return self.coco.loadImgs(img_id)[0]["file_name"]

    def create_class_map(self, min_nb_images):
        """
        Create a map from category_id: category_id, but if num_images is lower, then replace by one
        higher than the class_count. All classes with less than min_nb_images are mapped to the same class.
        """
        class_map = {}
        idx_to_name = {}
        class_counts = get_class_counts(self.root.rsplit('/', maxsplit=1)[0]) # = both train and val counts

        cls_above_min = sum(1 for c in class_counts.values() if c >= min_nb_images)
        cls_counter = 1

        for c_id, c_dict in self.coco.cats.items():
            class_c = class_counts[c_dict['name']]
            if class_c >= min_nb_images:
                class_map[c_id] = cls_counter
                idx_to_name[cls_counter] = self.coco.cats[c_id]['name']
                cls_counter += 1
            else:
                class_map[c_id] = cls_above_min
        idx_to_name[cls_above_min] = 'other'
        return class_map, idx_to_name

    def name_to_idx(self, name):
        for k, v in self.idx_to_name.items():
            if v == name:
                return k
        return None

    def get_num_classes(self):
        # All classes plus background
        return len(set(self.class_map.values())) + 1


def create_coco_dataset(split='train'):
    """
    Creates COCO style annotations which can be used for CuneiformDetection. Expects either train_tablets.txt or
    val_tablets.txt in the current directory. Should be cleaned up a little.
    :param split: train or val.
    """
    root = '../data'
    images, annotations, categories = [], [], []

    all_tablets = []
    with open(f'{split}_tablets.txt', 'r') as f:
        for line in f.readlines():
            all_tablets.append(line.strip())

    signs = pd.read_csv(os.path.join(root, "sign_map.csv"), sep=';')
    cat_id_map = {c: i for i, c in enumerate(sorted(signs['most_common'].values))}
    sign_map = load_sign_map(root)

    for k, v in cat_id_map.items():
        categories.append({'id': v, 'name': k})

    annot_id = 0
    annot_dir = f'{root}/annotations'
    annot_files = sorted(os.listdir(annot_dir))

    for img_id, annot in enumerate(annot_files):
        tablet_name, face = annot.rsplit('_', maxsplit=1)
        face = face.split('.')[0]

        if not tablet_name in all_tablets:
            continue
        else:
            print(tablet_name)

        img_name = f'{tablet_name}/{{}}_{face}.jpg'
        img_file = f'{root}/images/{img_name}'
        annot_file = f'{root}/annotations/{tablet_name}_{face}.json'

        # TODO: why are we using opencv here instead of PIL as in the other parts of the code?
        img = cv2.imread(img_file.format('SketchB'))  # Use sketchB to get format
        height, width, _ = img.shape

        file_name = img_name.format('TYPE')
        date_captured = "1970-01-01 00:00:00"

        with open(annot_file, 'r') as f:
            data = json.load(f)

        # There are some sides that do not have a single detection. That's a little bit annoying for some
        # IOU crop augmentations, so I'll leave them out of the dataset for now.
        if len(data) > 0:
            image_info = {'id': img_id, 'width': width, 'height': height, 'file_name': file_name,
                          'date_captured': date_captured}
            images.append(image_info)
        else:
            continue

        for j, key in enumerate(data):
            cat = get_target(data[key], sign_map=sign_map)
            try:
                cat_id = cat_id_map[cat]
            except KeyError:
                print(cat)
                continue # Weird categories. They're not that many so it may be okay to not have to use this?

            svg_str = data[key]['target']['selector']['value']
            try:
                coordinates = extract_points_from_svg(svg_str, return_int=True)
            except ValueError:
                continue # Circles don't need to be in the dataset here

            bbox = [int(i) for i in polygon_to_bbox(coordinates).flatten()]
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]

            # Crowd and area are just because pycocotools is a sh*tty piece of software and does not work without it
            # id needs to be a number, so include annot_id to not break backwards connection
            annot_info = {'id': annot_id, 'annot_id': key, 'image_id': img_id, 'category_id': cat_id,
                          'bbox': bbox, 'iscrowd': 0, 'area': 1.0}
            annotations.append(annot_info)
            annot_id += 1


    timestamp = time.strftime('%m%d_%H%M%S')
    dataset = {
        'info': {'Description': "Cuneiform Det V0", 'Creation': timestamp},
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    with open(f'detection_{split}_annot.json', 'w') as f:
        json.dump(dataset, f)


def create_train_val_split_detection():
    """
    This is a very ugly function to create a list of train and val images that are as balanced as they can be.
    The separate the kinds of tablets first (CUNES and HS), then orders them by the number of signs and selects
    a tablet from evenly divided groups. This selection is done such that the training and test samples are at
    the desired ratio. Then the counts of individual signs are counted and the process repeated to find a
    distribution as good as possible. Should probably clean this up if this will be distributed.
    """
    root = '../data_vat_o'
    all_tablets = sorted([t for t in os.listdir(os.path.join(root, 'images')) if not re.match(r'O_\d+', t)])
    all_annotations = sorted(os.listdir(os.path.join(root, 'annotations')))

    signs_per_tablet = []

    for tablet in all_tablets:
        tablet_annot = filter(lambda x: tablet in x, all_annotations)
        num_signs = 0

        for ta in tablet_annot:
            annot_file = f'{root}/annotations/{ta}'
            with open(annot_file, 'r') as f:
                data = json.load(f)
            num_signs += len(data)

        signs_per_tablet.append((tablet, num_signs))

    signs_per_tablet = sorted(signs_per_tablet, key=lambda x: x[1], reverse=True)
    hs_tablets = list(filter(lambda x: 'HS' in x[0], signs_per_tablet))
    cunes_tablets = list(filter(lambda x: 'CUNES' in x[0], signs_per_tablet))
    vat_tablets = list(filter(lambda x: 'VAT' in x[0], signs_per_tablet))

    tablet_lists = [cunes_tablets, hs_tablets, vat_tablets]

    test_frac = 0.2
    tol = 0.0025
    meta_data = pd.read_csv(f'{root}/sign_meta_data.csv', sep=';')
    start_seed = 1997
    best_i, best_error = 0, 1000

    for i in range(34, 35):
        test_tablets = []
        seed = start_seed + i * 1000
        random.seed(seed)

        for sublist in tablet_lists:
            best_ratio = 1.0
            tries = 0

            num_test = np.rint(int(test_frac * len(sublist))).astype(int)
            subsample_len = np.floor(len(sublist) / num_test).astype(int)

            while (best_ratio < (test_frac - tol) or (test_frac + tol) < best_ratio) and tries < 1000:
                tablets = [sublist[i:i+subsample_len] for i in range(0, len(sublist), subsample_len)]
                sub_test_tablets = [random.choice(t) for t in tablets]

                test_signs = sum(t[1] for t in sub_test_tablets)
                all_signs = sum(t[1] for t in sublist)

                # print(test_signs, all_signs - test_signs, (test_signs / all_signs))

                best_ratio = test_signs / all_signs
                tries += 1
            else:
                test_tablets.extend(sub_test_tablets)

        train_tablets = list(set(signs_per_tablet) - set(test_tablets))

        test_counts = Counter()
        train_counts = Counter()

        for t in test_tablets:
            tablet_counts = Counter(count_sings_per_tablet(t[0], meta_data=meta_data))
            test_counts += tablet_counts

        for t in train_tablets:
            tablet_counts = Counter(count_sings_per_tablet(t[0], meta_data=meta_data))
            train_counts += tablet_counts

        train_counts = sorted(((t, train_counts[t]) for t in train_counts), key=lambda x: x[1], reverse=True)
        train_counts = filter(lambda x: x[1] >= 20, train_counts)

        ratios = [(t[0], test_counts[t[0]] / t[1]) for t in train_counts]
        names, values = zip(*ratios)

        error_score = sum((v - 0.2) ** 2 for v in values)

        if error_score < best_error:
            best_error, best_i  = error_score, i

        print(f"{i}: {error_score:.3f} | {best_error:.3f} ({best_i})")


    train_tablet_names, _ = zip(*train_tablets)
    test_tablet_names, _ = zip(*test_tablets)

    train_tablet_names, test_tablet_names = sorted(train_tablet_names), sorted(test_tablet_names)

    with open(f'{root}/train_tablets.txt', 'w') as f:
        for t in train_tablet_names:
            f.write(t + "\n")

    with open(f'{root}/val_tablets.txt', 'w') as f:
        for t in test_tablet_names:
            f.write(t + "\n")

    x = np.arange(len(ratios))

    fig, axes = plt.subplots(1, 1, figsize=(12, 3.5))
    plt.bar(x, values, zorder=10, color='#7a2d9a')

    # plt.xticks(x, names, rotation=90)
    plt.hlines(0.2, 0, x[-1], color='#FFBE0B', lw=2.5, zorder=15)
    plt.grid(axis='y', zorder=0)

    plt.title("Fraction of validation examples per sign")
    plt.tight_layout()
    plt.savefig('./graphs/distribution_test_detection.png', dpi=300)
    plt.show()

    print(len(test_tablets), len(train_tablets), len(signs_per_tablet))

