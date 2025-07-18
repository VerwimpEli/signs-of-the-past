import re
import os
import json
from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd

import torch

def get_annotations(root: str) -> Sequence[str]:
    return sorted(os.listdir(os.path.join(root, 'annotations')))


def extract_points_from_svg(svg_string: str, return_int: bool = False):
    """
    Return list of coordinates based on svg XML.
    """

    points = re.findall(r'points=".*"', svg_string)
    if len(points) > 1:
        raise ValueError(f"Svg string contains more than one set of points: {svg_string}")
    if len(points) == 0:
        raise ValueError(f"Svg string does not contain points: {svg_string}")

    points = re.findall(r'[-\d.]+,[-\d.]+', points[0])
    points = [[float(f) for f in p.split(',')] for p in points]

    if return_int:
        return np.array(points, np.int32)
    else:
        return points

def extract_target(annotation) -> str | None:
    for body_part in annotation["body"]:
        if body_part['purpose'] == 'Transliteration':
            return body_part['value']
    return None

def extract_annotated_face(annotation) -> str | None:
    for body_part in annotation["body"]:
        if body_part['purpose'] == 'TabletSide':
            return body_part['value']
    return None

def extract_tablet_dimensions(annotation) -> tuple[int | None, int | None]:
    for body_part in annotation["body"]:
        if 'dimensions' in body_part:
            x, y = body_part['dimensions']['x'], body_part['dimensions']['y']
            return int(x), int(y)
    return None, None

def extract_line_and_char_idx(annotation) -> tuple[int, int]:

    line, char_idx = -1, -1
    for body_part in annotation["body"]:
        if body_part['purpose'] == 'Line':
            try:
                line = int(body_part['value'])
            except ValueError:
                # raise ValueError(f"Invalid line value '{body_part['value']}'")
                continue
        elif body_part['purpose'] == 'Charindex':
            try:
                char_idx = int(body_part['value'])
            except ValueError:
                # raise ValueError(f"Invalid Charindex value: '{body_part['value']}'")
                continue
    return line, char_idx



def get_target(annotation=None, sign_map=None, target=None) -> str:
    """
    Extract target from the annotation and map to the correct sign. Alternatively, a 'raw' target
    can directly be passed, and it will be transformed in the same way.

    """
    if target is None:
        target = extract_target(annotation)
        if target is None:
            return '?'

    # This conversion should be done because dropping {} and lowercase would make sign i, which is different
    if target == '{I}':
        return 'disz'

    # Drop determinative indicators
    target = target.replace('{', '')
    target = target.replace('}', '')

    # Don't use slashes, they don't allow using target names as folders
    target = target.replace("/", "-")

    # The distinction between upper and lower case is not important for sign recognition at this stage.
    target = target.lower()

    if sign_map is not None:
        try:
            target = sign_map[target]
        except KeyError:
            pass

    return target


def get_transliteration(root, annot_name, raw=False):
    sign_map = load_sign_map(root)

    with open(os.path.join(root, 'annotations', annot_name), 'r') as f:
        data = json.load(f)

    all_signs = defaultdict(list)
    for key, annot_value in data.items():
        try:
            line, char_idx = extract_line_and_char_idx(annot_value)
        except ValueError as e:
            line, char_idx = -1, -1
        target = extract_target(annot_value) if raw else get_target(annot_value, sign_map)
        # There are too many locations with more than one sign (wrongly), but to not lose them keep a list
        all_signs[line, char_idx].append(target)

    return all_signs


def polygon_to_bbox(coordinates: Sequence[float | int]) -> np.ndarray:
    """
    Extracts smallest bounding box around the polygon, returns as top-left and bottom-right coordinates
    :param coordinates:
    """
    top_left = np.min(coordinates, axis=0)
    bottom_right = np.max(coordinates, axis=0)
    return np.array([top_left, bottom_right])


def crop_img_with_bbox(img: np.ndarray, bbox: np.ndarray, square=True) -> np.ndarray:
    """
    Crop image with bounding box. If square = True, the rectangle will be turned into a square using the longest
    side. When the square goes over the edge, it is shifted to fit on the image (so in those cases the sign may not
    be completely centered)
    """
    img = img.copy() # Copy image here, if not the original array will not be clean anymore.
    bbox = np.array(bbox, np.int32)

    if not square:
        return img[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]
    else:
        height = bbox[1, 1] - bbox[0, 1]
        width = bbox[1, 0] - bbox[0, 0]
        cy = (bbox[1, 1] + bbox[0, 1]) // 2
        cx = (bbox[1, 0] + bbox[0, 0]) // 2

        if width < height:
            width = height
        else:
            height = width

        x0, x1 = cx - width // 2, cx + width // 2
        y0, y1 = cy - height // 2, cy + height // 2

        if x0 < 0:
            x1 += abs(x0)
            x0 = 0
        if y0 < 0:
            y1 += abs(y0)
            y0 = 0
        if x1 > img.shape[1]:
            x0 -= (x1 - img.shape[1])
            x1 = img.shape[1]
        if y1 > img.shape[0]:
            y0 -= (y1 - img.shape[0])
            y1 = img.shape[0]
        return img[y0:y1, x0:x1]


# TODO: not sure if this is the best place to have this. It depends on having a file there. Perhaps it is best
# to link it to a resource folder somewhere?
def load_sign_map(root, file_name: str = 'sign_map.csv'):
    df = pd.read_csv(os.path.join(root, file_name), sep=';')

    sign_map = {}

    for name, row in df.iterrows():
        readings = row['readings'].split(' ')
        most_common = row['most_common']
        for r in readings:
            sign_map[r] = most_common

    return sign_map


# TODO: same as above, depends on sign_meta_data existing.
def count_sings_per_tablet(tablet, meta_data=None):
    if meta_data is None:
        meta_data = pd.read_csv('../data/sign_meta_data.csv', sep=';')
    meta_data = meta_data[meta_data['tablet'] == tablet]
    target_counts = meta_data['target'].value_counts()
    return target_counts.to_dict()


# TODO: same as above, depends on sign_map existing.
def get_class_counts(root):
    """
    Returns class counts based on the sign map, which is expected to be found one level above root. This may not
    be the effective number of images as this doesn't take into account missing signs in some colors, but this way
    it remains consistent for every type of sign.

    Be weary: this returns a default dict, so for classes that aren't in the sign map it returns zero
    """
    sign_map = pd.read_csv(os.path.join(root, "./sign_map.csv"), sep=';')
    class_counts = {name: count for name, count in sign_map[["most_common", "num_signs"]].values}
    class_counts = defaultdict(lambda: 0, class_counts)
    return class_counts


def get_dataset_stats(loader: torch.utils.data.DataLoader, num_samples = 500):
    """
    Very slow function to estimate the mean and standard deviation of a dataset.
    """
    class ImgStatEstimator:
        def __init__(self):
            self.count = 0
            self.mean = 0
            self.M2 = 0

        def update(self, new_value):
            self.count += 1
            delta = new_value - self.mean
            self.mean += delta / self.count
            delta2 = new_value - self.mean
            self.M2 += delta * delta2

        def finalize(self):
            if self.count < 2:
                return float("nan")
            else:
                return self.mean, self.M2 / self.count, self.M2 / (self.count - 1)


    channel_est = [ImgStatEstimator(), ImgStatEstimator(), ImgStatEstimator()]
    for s, (data, target) in enumerate(loader):
        print(s, end=', ', flush=True)
        data = data.squeeze()
        data = data.view(3, -1)
        for i, channel in enumerate(data):
            for value in channel:
                channel_est[i].update(value)

        if s > num_samples:
            print()
            break

    for ce in channel_est:
        print(ce.finalize())

