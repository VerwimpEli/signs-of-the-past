import json
import os.path
import time
import urllib.request
from urllib.error import HTTPError
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import shapely
from PIL import Image

from .utils import (get_annotations, polygon_to_bbox, load_sign_map, extract_points_from_svg, extract_annotated_face,
                    extract_tablet_dimensions, get_target, crop_img_with_bbox, extract_line_and_char_idx)


ALL_IMG_TYPES = ['sketchB', 'sketchA', 'normal', 'Sketch01Hard', 'colorA', 'colorB', 'colorC', 'colorD',
                 'colorE', 'colorF', 'colorE', 'colorG', 'colorH']

class SignExtract:
    """
    Convenience class during the creation of sign crops, which makes it easier to calculate areas and overlap between signs
    """
    def __init__(self, tablet: str,
                 target: str,
                 coords: Sequence[float | int],
                 visual_face: str,
                 annotated_face: str = None,
                 id_str: str = '',
                 tablet_dim=None,
                 line_idx: int = None,
                 char_idx: int = None
                 ):
        """
        :param tablet: the name of the tablet this sign belongs to.
        :param target: target name of this sign.
        :param coords: list of coordinates that define the polygon.
        :param visual_face: the face of the tablet on which the annotation is made (i.e., the annotation file name).
        :param annotated_face: the face of the tablet to which the sign belongs (this is indicated inside the specific
                               annotation for the sign)
        :param id_str: the id string inside the annotation file. These are not globally unique somehow.
        :param tablet_dim: (width, height) of the tablet the sign belongs to. Used to get the center of the sign.
        """
        self.tablet = tablet
        self.target = target
        self.coords = coords
        self.polygon = shapely.Polygon(coords)
        self.visual_face = visual_face
        self.annotated_face = annotated_face if annotated_face is not None else visual_face
        self.id_str = id_str
        self.tablet_dim = tablet_dim    # width then height, this is not safe
        self.line_idx = line_idx
        self.char_idx = char_idx

    def get_name(self) -> str:
        # Assuming that the last part of the id is unique in a single tablet annotation. It's almost unique
        # over the entire dataset, except for 3 ids that are completely equal.
        return f"{self.tablet}_{self.visual_face}_{self.id_str.split('-')[-1]}"

    def get_area(self, bbox=False) -> float:
        """
        Calculate the area of the sign, either of the polygon or the surrounding bounding box.
        :param bbox: if true, return area of bounding box.
        """
        if bbox:
            bbox = polygon_to_bbox(np.array(self.polygon.exterior.coords))
            return (bbox[0, 0] - bbox[1, 0]) * (bbox[0, 1] - bbox[1, 1])  # type: ignore
        else:
            return self.polygon.area

    def intersection_area(self, other) -> float:
        """
        Calculate the intersection area between self another sign extract. Shapely sometimes returns a
        GEOS Exception, which is caught here as I don't really know why it happens. It doesn't happen
        too often. This method returns 0.0 in that case, which may lead to an underestimate of how much overlap exists.
        :return: overlap area in whatever coordinate system the polygon is in.
        """
        try:
            if (self.tablet == other.tablet and self.visual_face == other.visual_face and self.id_str != other.id_str
                    and self.polygon.intersects(other.polygon)):
                    return self.polygon.intersection(other.polygon).area
            else:
                return 0.0
        except shapely.errors.GEOSException:
            return 0.0   # not the best solution because if it does have an overlap,
                         # this may underestimate the total overlap

    def get_total_overlap(self, others: Sequence) -> float:
        """
        Returns an estimate of how much this sign is covered by other signs. This is an estimate because two signs
        overlapping the same part of this sign are not considered (i.e., the total overlap can be larger than 1.0).
        :return: an estimate of the overlap divided by the area of the sign.
        """
        total_overlap = 0.0
        for other in others:
            total_overlap += self.intersection_area(other)
        return total_overlap / self.get_area()

    def get_relative_center(self) -> Tuple[float, float]:
        """
        :return: the relative center of mass of the sign inside the tablet face.
        """
        cx, cy = self.polygon.centroid.x, self.polygon.centroid.y

        if self.tablet_dim[0] is not None:
            return cx / self.tablet_dim[0], cy / self.tablet_dim[1]
        else:
            return np.NAN, np.NAN

    def __repr__(self) -> str:
        return f"{self.target} ({self.id_str:03d}) {self.polygon}"


def verify_annotations(root: str):
    """
    Verifies that the annotation folder is where it should be.
    """
    annot_path = os.path.join(root, 'annotations')
    if os.path.exists(annot_path):
        annot_files = os.listdir(annot_path)
        print(f"Found {len(annot_files)} annotations files")
    else:
        raise ValueError(f"No folder called 'annotations' found in {root}.")


def verify_sign_map(root: str):
    """
    Verifies that the sign map is where it should be and some basic check that it is correct. It could be created
    automatically, but it shouldn't change, so for consistency it is better to keep the exact same one everywhere.
    """
    sign_map_path = os.path.join(root, 'sign_map.csv')
    if os.path.exists(sign_map_path):
        sign_map = pd.read_csv(sign_map_path, sep=';')
        assert 'readings' in sign_map.columns, "No column 'reading' in sign map"
        assert 'most_common' in sign_map.columns, "No column 'most_common in sign map"
    else:
        raise ValueError(f"Did not find a file called sign_map.csv in {root}")


def check_or_download_images(root: str, img_types: Sequence[str]):
    annot_files = sorted(get_annotations(root))

    all_links = []
    annot_without_links = 0

    # Find all urls in the annotation files
    for annot in annot_files:
        annot_file = os.path.join(root, 'annotations', annot)

        with open(annot_file, 'r') as f:
            data = json.load(f)

        annot_links = []
        for key in data:
            link = data[key]['target']['source']
            # Find the image type that was annotated and replace it by the generic 'TYPE'
            for it in ALL_IMG_TYPES:
                if it in link:
                    link = link.replace(it, 'TYPE')
                    break
            annot_links.append(link)

        annot_links = set(annot_links)
        if len(annot_links) > 0:
            all_links.extend(annot_links)
        else:
            annot_without_links += 1

    all_links = sorted(list(set(all_links)))
    print(f"Found {len(all_links)} unique URLs and {annot_without_links} annotations without URLs.")

    # Download all images and put them in the correct place.
    for full_url in all_links:
        # Tablet name is the last folder before the file name
        tablet_name = full_url.split(':')[-2]

        if 'HS' in tablet_name:
            # The annotations have HS_ while the images of HS. We stick to the naming of the annotations.
            tablet_name = tablet_name.replace('HS', 'HS_')
        elif 'O.0' in tablet_name:
            # The annotations have O_ instead of O.0, replace.
            tablet_name = tablet_name.replace('O.0', 'O_')

        img_dir = f'{root}/images/{tablet_name}'
        os.makedirs(img_dir, exist_ok=True)

        # The face is after the last underscore in the file name, which comes after the last ':', but before the
        # next '/'.
        details = full_url.split(':')[-1].split('/')[0].split('_')

        if len(details) > 2:
            # Assume that if there are more than two underscores, the last one is the face. This is not
            # guaranteed to be true at all, but for now it works. In theory, there could be a tablet VAT00782_E_SketchB,
            # which wouldn't be counted.
            face = details[-1]
        else:
            # Fat cross format is not explicitly added in the URLs, but we want to add it here
            # for consistency so that all images have a 'face'.
            face = 'fc'

        for it in img_types:
            # Assume the images are all JPGs this may change when other file types are used.
            img_local_path = f'{img_dir}/{it}_{face}.jpg'

            if 'O_' in tablet_name and it.lower() == 'sketchb':
                # For O_ tablets, SketchB is called Sketch01Hard
                it = 'Sketch01Hard'

            typed_url = full_url.replace('TYPE', it)

            if not os.path.exists(img_local_path):
                print(f"Downloading: \t {typed_url}")
                try:
                    with urllib.request.urlopen(typed_url) as f:
                        img = f.read()
                    with open(img_local_path, 'wb') as g:
                        g.write(img)
                except urllib.error.HTTPError as e:
                    print(e)
                    print(f"Image not found. Some colors are not available for some tablets. Continuing...")



def build_sign_objects(root: str, annot: str, sign_map: dict[str, str], max_overlap: float = 0.9)\
        -> Sequence[SignExtract]:
    """
    Builds a list of SingExtracts for a single annotation file.

    :param root: root of the dataset.
    :param annot: name of the json-annotation file.
    :param sign_map: sing_map for the conversions of targets.
    :param max_overlap: the maximal overlap between two annotated signs. If higher, one of them will be excluded.
    :return: list of SignExtracts for the tablet face of annot.
    """
    annot_path = os.path.join(root, 'annotations', annot)
    with open(annot_path, 'r') as f:
        data = json.load(f)

    tablet_name, face = annot.rsplit('_', maxsplit=1)
    face = face.split('.')[0]

    sign_objects  = []

    # Create sign objects and meta-data
    for i, key in enumerate(data):
        # Get points that define the polygon annotation
        svg_str = data[key]['target']['selector']['value']
        try:
            points = extract_points_from_svg(svg_str)
        except ValueError:  # Either missing points or annotation as circle, skip these.
            continue

        # Get face to which the annotation belongs, this is sometimes different from the actual annotation.
        annotated_face = extract_annotated_face(data[key])

        # Get the width and height of tablet
        tablet_width, tablet_height = extract_tablet_dimensions(data[key])

        # Get target, but remap with sign_map so that all readings of the same sign have the same target
        target = get_target(data[key], sign_map=sign_map)

        # Extract line and char idx
        line_idx, char_idx = extract_line_and_char_idx(data[key])

        # Create the sign object
        sign = SignExtract(tablet_name, target, points, face, annotated_face, key,(tablet_width, tablet_height),
                           line_idx, char_idx)

        # Some signs are annotated more than once, filter them here with max_overlap (with a single, other sign)
        area = sign.get_area()
        overlaps = [sign.intersection_area(other) / area for other in sign_objects]
        if len(sign_objects) == 0 or max(overlaps) <= max_overlap:
            sign_objects.append(sign)

    return sign_objects


def img_type_conversion(img_type, tablet_name):
    """
    HS and CUNES are named wrongly, so img_type needs to be converted when making crops for these images.

    Note: it would also have been possibly to rename the images, but then the image names would be different locally
    and on the IIIF server. This would be confusing and re-downloading everything when something is unsure is more
    wasteful than remaking all crops.
    """
    conv_table = {
        'ColorH': 'ColorA',
        'ColorA': 'ColorB',
        'ColorB': 'ColorC',
        'ColorC': 'ColorD',
        'ColorD': 'ColorE',
        'ColorE': 'ColorF',
        'ColorF': 'ColorG',
        'ColorG': 'ColorH',
    }

    if 'HS' in tablet_name or 'CUNES' in tablet_name:
        try:
            img_type = conv_table[img_type]
        except KeyError:
            img_type = img_type

    return img_type


def create_crops(root: str, img_types: Sequence[str], store_crops = True, rebuild_meta_data = True,
                 max_overlap: float = 0.9, correct_old_namings = True):
    """
    Create the crops and the accompanying metadata for the classification datasets.
    :param root: root of the dataset
    :param img_types: img_types to make crops for
    :param store_crops: actually store crops, if False, only metadata is created
    :param rebuild_meta_data: if True, existing metadata will be completely overwritten (which is probably best to do always)
    :param max_overlap: maximal overlap between two annotated signs, to filter out duplicates
    :param correct_old_namings: if true, will change colors of HS and Cunes with one step (i.e. ColorA -> ColorB, IIIF links are still wrong)
    """
    annotations = get_annotations(root)
    sign_map = load_sign_map(root)

    meta_data_path = os.path.join(root, 'sign_meta_data.csv')
    if os.path.exists(meta_data_path) and not rebuild_meta_data:
        current_meta_data = pd.read_csv(meta_data_path, sep=';', index_col='name')
    else:
        current_meta_data = None
        with open(meta_data_path, 'w') as f:
            f.write("name;tablet;sign_id;target;visual_face;annotated_face;poly_area;bbox_area;overlap;cx;cy;"
                    "line_idx;char_idx\n")

    for annot in annotations:
        print(f"Cropping: \t {annot}")
        tablet_name, face = annot.rsplit('_', maxsplit=1)
        face = face.split('.')[0]

        tablet_face_signs = build_sign_objects(root, annot, sign_map, max_overlap)

        for img_type in img_types:
            if correct_old_namings:
                img_file_type = img_type_conversion(img_type, tablet_name)
            else:
                img_file_type = img_type
            img_file = f'{root}/images/{tablet_name}/{img_file_type}_{face}.jpg'

            # Special case for pngs of normal maps
            # img_file = f'{root}/images/{tablet_name}/{tablet_name}_{img_type}_{face}.png'

            if os.path.exists(img_file):
                img = np.asarray(Image.open(img_file))
            else:
                print(f"Image '{img_file}' not found. Continuing...")
                continue

            if store_crops:
                for sign in tablet_face_signs:
                    bbox = polygon_to_bbox(sign.coords)
                    sign_extract = crop_img_with_bbox(img, bbox, square=True)

                    sign_dir = f'{root}/sign_crops/{sign.target}'
                    os.makedirs(sign_dir, exist_ok=True)
                    img_name = f"{sign.get_name()}_{img_type}.jpg"

                    Image.fromarray(sign_extract).save(os.path.join(sign_dir, img_name))

        with open(f'{root}/sign_meta_data.csv', 'a') as f:
            for sign in tablet_face_signs:
                sign_key = f"{sign.target}/{sign.get_name()}"
                if current_meta_data is None or sign_key not in current_meta_data.index:
                    info_str = (f"{sign.target}/{sign.get_name()};{sign.tablet};{sign.id_str};{sign.target};"
                                f"{sign.visual_face};{sign.annotated_face};{sign.get_area():.2f};"
                                f"{sign.get_area(bbox=True):.2f};{sign.get_total_overlap(tablet_face_signs)};"
                                f"{sign.get_relative_center()[0]};{sign.get_relative_center()[1]};"
                                f"{sign.line_idx};{sign.char_idx}")
                    f.write(info_str + "\n")


def verify_train_val_split(root: str, split_file_name: str, train_frac: float = 0.8):
    """
    Verify the train-val split by comparing it to the current meta-data list. They need to include the same crop names.
    If there are old signs they are removed from the split_file, if there are new ones they are assigned to either
    train or val based on the current ratio of train and val samples. If a new one is created, it is not overwritten
    but written to a new file. It is best to keep old files, as old versions of the dataset may sometimes be necessary.

    :param root: root of the dataset
    :param split_file_name: file_name of the split file.
    :param train_frac: fraction of the training set. WARNING: if a new train_frac is used, this will try to correct
    that file to the new ratio. In such cases, a new split file should be made instead.
    """
    split_file_path = os.path.join(root, split_file_name)
    if not os.path.exists(split_file_path):
        raise ValueError(f"Split file {split_file_path} not found.")

    meta_data = pd.read_csv(os.path.join(root, 'sign_meta_data.csv'), sep=';', index_col='name')
    meta_data_names = set(meta_data.index)

    split_file = pd.read_csv(split_file_path, sep=' ', header=None, names=['name', 'train', 'val', 'test'])
    split_file['mod_name'] = split_file['name'].apply(lambda x: x.rsplit('_', maxsplit=1)[0])
    split_file = split_file.set_index('mod_name')
    split_file_names = set(split_file.index)

    missing_in_meta = list(split_file_names - meta_data_names)
    missing_in_split = list(meta_data_names - split_file_names)

    if len(missing_in_split) > 0 or len(missing_in_meta) > 0:
        # Add new sings, a little involved as we need to keep track of the current train/val split for each class
        split_file['target'] = split_file['name'].apply(lambda x: x.split('/', maxsplit=1)[0])
        train_val_counts = pd.pivot_table(split_file, values=['train', 'val'], index='target',
                                          aggfunc={'train': ['mean', 'count'], 'val': 'mean'})

        new_signs = []
        for i, sign_name in enumerate(missing_in_split):
            target = sign_name.split("/", maxsplit=1)[0]
            sign_name += '_TYPE.jpg'

            try:
                target_train_frac = train_val_counts.loc[target, ('train', 'mean')]
                count = train_val_counts.loc[target, ('train', 'count')]

                if target_train_frac >= train_frac:
                    new_signs.append([sign_name, 0, 1, 0])
                    target_train_frac = target_train_frac * (count / (count + 1))
                else:
                    target_train_frac = (target_train_frac +  1 / count) * (count / (count + 1))
                    new_signs.append([sign_name, 1, 0, 0])

                train_val_counts.loc[target] = [count + 1, target_train_frac, 1.0 - target_train_frac]

            except KeyError:  # No sign of this category was yet in the current split file, add as a val sample
                new_signs.append([sign_name, 0, 1, 0])
                train_val_counts.loc[target] = [1, 0.0, 1.0]

        # Drop old signs that are still in the split file
        split_file = split_file.drop(missing_in_meta, axis=0)

        split_file = split_file[['name', 'train', 'val', 'test']]
        split_file = pd.concat([split_file, pd.DataFrame(new_signs, columns=['name', 'train', 'val', 'test'])])
        split_file = split_file.set_index('name')

        timestamp = time.strftime('%m%d_%H%M')
        split_file.to_csv(os.path.join(root, f"{split_file_name[:-4]}_{timestamp}.txt"), sep=' ', header=False)

        print(f"Found {len(missing_in_meta)} old sings in the split file, they are removed.")
        print(f"Found {len(missing_in_split)} new signs not in the split file, they are added.")
    else:
        print("No differences found between current signs and split file.")


def create_sign_split_from_coco(root):
    """
    Alternative split creation that completely separates tablets. It expects to have a val_tablets.txt and
    train_tablets.txt (which are created when the detection tablets are split) and then uses the meta-data file to
    create a sign list which is stored as coco_tablet_split.txt
    :return:
    """
    sign_meta = pd.read_csv(os.path.join(root, 'sign_meta_data.csv'), sep=';')
    train_tablets = pd.read_csv(os.path.join(root, 'coco_annotations', 'train_tablets.txt'), header=None,names=['tablet_name'])
    val_tablets = pd.read_csv(os.path.join(root, 'coco_annotations', 'val_tablets.txt'), header=None, names=['tablet_name'])

    sign_meta['name'] += '_TYPE.jpg'
    sign_meta['train'] = sign_meta['tablet'].isin(train_tablets['tablet_name']).astype(int)
    sign_meta['val'] = sign_meta['tablet'].isin(val_tablets['tablet_name']).astype(int)
    sign_meta['test'] = 0

    split_file = sign_meta[['name', 'train', 'val', 'test']]
    print(split_file)
    split_file.to_csv(os.path.join(root, f"coco_tablet_split.txt"), sep=' ', header=False, index=False)



def create_sign_crops(root: str, img_types: Sequence[str], check_or_download_img = True, make_crops = True,
                      verify_train_val = True, store_crops = True):
    """
    This method will create all the necessary files for the classification dataset. It expects three files to be
    files to be there in advance: a folder with JSON annotation, the sign map as 'sign_map.csv' and, although not
    strictly necessary, a split_file with the name equal to the parameter below.

    Then, this will make the cut-outs. They are completely remade every time this is run, as it is easier than checking
    whether something has changed on the annotation. Together with this, the metadata is build, which is stored in
    the root directory as sign_meta_data.csv. Old sign crops are not removed, as they may be useful for old dataset
    split files. But in general, they could be removed.

    Finally, the split file is updated by removing old signs that are no longer in the metadata and including new ones,
    and assigning a train val split to them.
    """
    verify_annotations(root)
    verify_sign_map(root)

    if check_or_download_img:
        check_or_download_images(root, img_types)
    if make_crops:
        create_crops(root, img_types, store_crops=store_crops)
    if verify_train_val:
        verify_train_val_split(root, split_file_name='train_val_split_cunes_hs_vat_o.txt')

    # Alternative split file creation with tablets completely separated between train and val.
    # create_sign_split_from_coco(root)


if __name__ == '__main__':
    create_sign_crops(root='../data', img_types=['SketchB'])