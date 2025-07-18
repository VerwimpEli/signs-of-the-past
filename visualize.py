import argparse
import json
from symtable import Class

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle, Polygon, Circle

from src import *


def main_visualize():

    parser = argparse.ArgumentParser()
    parser.add_argument('quick', nargs='*')
    parser.add_argument('-t', '--tablet', default='CUNES51-01-07')
    parser.add_argument('-f', '--face', default='front')
    parser.add_argument('--type', default='SketchB')
    parser.add_argument('--poly', action='store_true')
    parser.add_argument('--signs', nargs='*')
    parser.add_argument('--line', type=int)
    parser.add_argument('--sign_id')
    args = parser.parse_args()

    if len(args.quick) > 0:
        print(args.quick)
        tablet_name, face = args.quick[0].rsplit('_', maxsplit=1)
        face = face.split('.')[0]
        if len(args.quick) > 1:
            args.sign_id = args.quick[1:]
            print(args.sign_id)
    else:
        tablet_name = args.tablet
        face = args.face

    tablet_name = tablet_name.replace('c', 'CUNES51-01-')
    tablet_name = tablet_name.replace('h', 'HS_')

    img_type = args.type

    root = '../data_vat_o'
    split_file = 'train_val_split_cunes_hs_vat_o.txt'

    annot_file = f'{root}/annotations/{tablet_name}_{face}.json'
    img_file = f'{root}/images/{tablet_name}/{img_type}_{face}.jpg'

    # pred_file = './results/37/0508_194637/predictions.csv'
    # predictions = pd.read_csv(pred_file, sep=';')
    # predictions = join_predictions_with_meta(predictions, root=root)
    #
    # predictions = predictions[predictions['tablet'] == tablet_name]
    # idx_to_class =  CuneiformClassification(root, min_nb_img=20, split='val', lazy_load=True, img_type="SketchB",
    #                                         split_file=split_file).idx_to_class

    with open(annot_file, 'r') as f:
        data = json.load(f)

    img = np.array(Image.open(img_file))
    width, height, _ = img.shape
    fig, ax = plt.subplots()

    sign_map = load_sign_map(root)

    for i, key in enumerate(data):
        if args.sign_id is None or key in args.sign_id:
            svg_str = data[key]['target']['selector']['value']

            try:
                coordinates = extract_points_from_svg(svg_str, return_int=True)
                bbox = polygon_to_bbox(coordinates)
                target = get_target(data[key], sign_map=sign_map)

                # target = extract_line_and_char_idx(data[key])
                # target = get_target(data[key])

                if args.signs is None or target in args.signs:

                    # try:
                    #     pred = predictions[predictions['sign_id'] == key]['p1']
                    #     pred = idx_to_class[pred.values[0]]
                    #
                    #     bw, bh = bbox[1, 0] - bbox[0, 0], bbox[1, 1] - bbox[0, 1]
                    #     color = 'limegreen' if pred == target else 'lightcoral'
                    #     text = pred if pred == target else f"{target}:{pred}"
                    #     ax.add_patch(Rectangle((bbox[0, 0], bbox[0, 1]), bw, bh, fill=False, lw=3, edgecolor=color))
                    #
                    #     if args.poly:
                    #         ax.add_patch(Polygon(coordinates, fill=False, lw=2, edgecolor='#ff0000'))
                    #
                    #     ax.text(bbox[0, 0], bbox[0, 1], text, color="black", backgroundcolor=color, va='bottom',
                    #             ha='left', fontsize=12)
                    #
                    # except IndexError:
                    #     continue

                    bw, bh = bbox[1, 0] - bbox[0, 0], bbox[1, 1] - bbox[0, 1]
                    ax.add_patch(Rectangle((bbox[0, 0], bbox[0, 1]), bw, bh, fill=False, lw=2, edgecolor='green'))
                    if args.poly:
                        ax.add_patch(Polygon(coordinates, fill=False, lw=1.5, edgecolor='#0000ff'))
                    ax.text(bbox[0, 0], bbox[0, 1], target, color="black", backgroundcolor='green', va='bottom', ha='left',
                            fontsize=12)

            except ValueError:
                # circle = extract_circle_from_svg(svg_str)
                # ax.add_patch(Circle(circle[0:2], radius=circle[2], fill=False, lw=3, edgecolor="#ff0000"))
                pass

    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main_visualize()