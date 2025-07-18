import json
import os
import re

import torch
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

from src import visualize_single_pred, load_sign_map
import src.utils as utils


def select_tablet(allowed_tablets, img_folder, img_type):
    allowed_proveniences = set([re.findall(r'[A-Za-z]+', t)[0] for t in allowed_tablets])
    allowed_proveniences = sorted(list(allowed_proveniences))

    input_str = "Available proveniences: "
    for i, ap in enumerate(allowed_proveniences):
        input_str += f"\n  ({i+1}): {ap}"
    input_str += '\nSelect provenience: '
    selected_prov = allowed_proveniences[int(input(input_str)) - 1]

    allowed_tablets = sorted([t for t in allowed_tablets if selected_prov in t])

    input_str = "Available tablets: "
    for i, at in enumerate(allowed_tablets):
        input_str += f"\n  ({i + 1}): {at}"
    input_str += '\nSelect tablet: '
    selected_tablet = allowed_tablets[int(input(input_str)) - 1]

    available_faces = sorted(os.listdir(os.path.join(img_folder, selected_tablet)))
    available_faces = [af for af in available_faces if img_type in af]

    input_str = "Available faces: "
    for i, af in enumerate(available_faces):
        input_str += f"\n  ({i + 1}): {af}"
    input_str += '\nSelect face: '
    selected_face = available_faces[int(input(input_str)) - 1]

    return os.path.join(img_folder, selected_tablet, selected_face)

def load_allowed_tablets(file: str):
    tablets = []
    with open(file, 'r') as f:
        for line in f.readlines():
            tablets.append(line.strip())
    return tablets

def load_idx_to_class(idx_to_class_path):
    idx_to_class = {}
    with open(idx_to_class_path, 'r') as f:
        for line in f.readlines():
            idx, cls = line.strip().split(' ')
            idx_to_class[int(idx)] = cls
    return idx_to_class

def class_tool_transforms(img_type, resize=224):

    mean, std = utils.get_image_statistics(img_type, log=True)

    base_transforms = [
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=resize, antialias=True),
        v2.Normalize(mean, std),
    ]

    return v2.Compose(base_transforms)


@torch.no_grad()
def main():

    print("Cuneiform classifier version 0.01. \n"
          "Select your tablet first. Then you can control the tool using: \n"
          "t: turn on rectangle selector \n"
          "r: reveal labels \n")

    # Data root directory with images and annotations folder
    data_root = '../data_vat_o'

    # Root of other files
    root = '../class_tool_files/misc'

    # File that lists all validation tablets.
    allowed_tablets_path = f'{root}/val_tablets.txt'

    # Path where the weights for the model are stored.
    # model_path = f'{root}/coco_vat_model.pth'
    model_path = f'output/models/model00.pth'

    # Folder where images are stored. They are expected to be e.g.
    # - images
    #   - CUNES51-01-01
    #       - ColorA_back.jpg
    #       - ColorA_bottom.jpg
    #       - ColorB_back.jpg
    #       - ...
    #   - CUNES51-01-02
    #   - CUNES51-01-03
    #   - ...
    img_folder = f'{data_root}/images'

    # Folder with annotation in root.
    annot_folder = f'{data_root}/annotations'

    # Image type. Model is trained on SketchB, so don't expect it to work on other images.
    img_type = 'SketchB'

    # idx_to_class path: this file has the mappings from output index to names of the signs
    idx_to_class_path = f'{root}/idx_to_class.txt'

    # END PARAMETERS

    # Select tablet through command line input.
    allowed_tablets = load_allowed_tablets(allowed_tablets_path)
    # image_path = select_tablet(allowed_tablets, img_folder, img_type)
    image_path = 'P387810.jpg'
    idx_to_class = load_idx_to_class(idx_to_class_path)

    sign_map = load_sign_map(root)
    annotations_visible = False

    image = Image.open(image_path)
    image_arr = np.array(image)

    # Pad image so that selection can be larger than the image itself
    pad_width = 500
    image_arr = np.pad(image_arr, [[pad_width, pad_width], [pad_width, pad_width], [0, 0]], mode='constant')

    # Load model
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()

    # Load transformations (normalization and resize)
    val_transform = class_tool_transforms(img_type, resize=224)

    fig, axd = plt.subplot_mosaic(
        [["image", "cutout"],
                ["image", "pred"]],
        layout="constrained", width_ratios=[1.2, 1.]
    )

    axd['image'].imshow(image)
    axd['image'].axis('off')

    def toggle_selector(event):
        """
        Sub-function to toggle rectangle selector (t) and revealing the annotations (r)
        """
        if event.key == 't':
            if selector.active:
                selector.set_active(False)
            else:
                selector.set_active(True)
        if event.key == 'r':
            toggle_annotations()

    def select_callback(e_click, e_release):
        """
        Callback of selector. It cuts out a square patch in the image and uses that as the input to
        the loaded model. The results are shown as a bar chart of the k-most likely predictions.
        """
        x1, y1 = int(e_click.xdata), int(e_click.ydata)
        x2, y2 = int(e_release.xdata), int(e_release.ydata)

        cy = (y1 + y2) // 2  + pad_width
        cx = (x1 + x2) // 2  + pad_width

        height = (x2 - x1) // 2
        width = (y2 - y1) // 2

        if height == 0 or width == 0:
            return

        # Make selection square, because our models are trained using squares.
        if height > width:
            width = height
        else:
            height = width

        img_cutout = image_arr[cy - height:cy + height, cx - width:cx + width]
        torch_img = torch.tensor(img_cutout).unsqueeze(0)
        torch_img = torch_img.permute(0, 3, 1, 2)

        transformed_img = val_transform(torch_img)
        output = model(transformed_img)
        confidence = torch.nn.functional.softmax(output, dim=1)

        k = 5
        top_output = torch.topk(confidence, k=k)
        conf_values = top_output.values[0].numpy()
        pred = top_output.indices[0].numpy()

        axd['cutout'].cla()
        axd['pred'].cla()

        visualize_single_pred(root=root, axes=[axd['cutout'], axd['pred']], pred=pred[::-1], conf=conf_values[::-1],
                              img=img_cutout, idx_to_class=idx_to_class, color='#7a2d9a')
        plt.draw()

    def add_annotations():
        """
        Adds the annotations to the image. Initially they are set to be non-visible, only after toggling 'r', they
        can become visible.
        """
        annotations = []
        tablet = image_path.split('/')[-2]
        face = image_path.split('_')[-1][:-4]
        annot_file = os.path.join(annot_folder, f"{tablet}_{face}.json")

        try:
            with open(annot_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            return []

        for key in data:
            svg_str = data[key]['target']['selector']['value']

            try:
                coordinates = utils.extract_points_from_svg(svg_str, return_int=True)
                bbox = utils.polygon_to_bbox(coordinates)

                target = utils.get_target(data[key], sign_map=sign_map)

                bw, bh = bbox[1, 0] - bbox[0, 0], bbox[1, 1] - bbox[0, 1]
                r = axd['image'].add_patch(Rectangle((bbox[0, 0], bbox[0, 1]), bw, bh, fill=False, lw=2,
                                                     edgecolor='limegreen'))
                t = axd['image'].text(bbox[0, 0], bbox[0, 1], target, color="black", backgroundcolor='limegreen',
                                      va='bottom', ha='left', fontsize=12)
                r.set_visible(False)
                t.set_visible(False)
                annotations.append((r, t))
            except ValueError:
                pass
        return annotations

    # all_annotations = add_annotations()
    all_annotations = []

    def toggle_annotations():
        # Toggles annotations on or off, depending on the current status
        nonlocal annotations_visible
        annotations_visible = not annotations_visible
        for r, t in all_annotations:
            r.set_visible(annotations_visible)
            t.set_visible(annotations_visible)
        plt.draw()

    selector = RectangleSelector(axd['image'], select_callback, useblit=True, button=[1], minspanx=10, minspany=10,
                                 interactive=True, drag_from_anywhere=True, ignore_event_outside=False)

    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()


