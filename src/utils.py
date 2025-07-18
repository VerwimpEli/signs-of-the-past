import json
import os
import re
import time
from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib import font_manager

def extract_circle_from_svg(svg_string: str):
    def extract_named_float(pattern, string):
        field = re.findall(pattern, string)[0]
        return float(re.findall(r"[\d.]+", field)[0])

    cx = extract_named_float(r'cx="[\d.]+', svg_string)
    cy = extract_named_float(r'cy="[\d.]+', svg_string)
    r = extract_named_float(r'r="[\d.]+', svg_string)

    return cx, cy, r

def pretty_print_transliteration(transliteration, fmt='7s'):
    num_row = max(co[0] for co in transliteration) + 1
    num_cols = max(co[1] for co in transliteration) + 1

    for i in range(1, num_row):
        print(f"{i:2d}: ", end=' ')
        for j in range(1, num_cols):
            text = transliteration[i, j]
            if len(text) == 0:
                text = '*'
            elif len(text) == 1:
                text = text[0]
            else:
                text = '/'.join(text)
            print(f"{text:{fmt}}", end=' ')
        print()

    print(f"Signs with no line: {''.join([f'{k} - {v}' for k, v in transliteration.items() if k[0] == -1])}")

def unicode_to_ascii_atf(string):
    unicode_ascii_translation = {
        "š": "sz",
        "Š": "SZ",
        "ṣ": "s,",
        "Ṣ": "S,",
        "ṭ": "t,",
        "Ṭ": "T,",
        "ś": "s'",
        "Ś": "S'",
        "ŋ": "j",
        "Ŋ": "J",
        "ḫ": "h,",
        "Ḫ": "H,",
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "ₓ": "x"
    }

    for u, a in unicode_ascii_translation.items():
        string = string.replace(u, a)

    return string

def ascii_to_unicode_atf(string):
    ascii_unicode_translation = {
        "sz": "š",
        "SZ": "Š",
        "s,": "ṣ",
        "S,": "Ṣ",
        "t,": "ṭ",
        "T,": "Ṭ",
        "s'": "ś",
        "S'": "Ś",
        "j": "ŋ",
        "J": "Ŋ",
        "h,": "ḫ",
        "H,": "Ḫ",
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
        "x": "ₓ"}

    for a, u in ascii_unicode_translation.items():
        string = string.replace(a, u)

    return string

def unicode_to_u_nb(unicode):
    return re.findall(r'U[\da-f]+', str(unicode.encode('unicode_escape')))[0]

def get_cuneiform_font_prop():
    font_path = '../NotoSansCuneiform-Regular.ttf'
    return font_manager.FontProperties(fname=font_path)

def load_cls_name_to_unicode_map(root, file_name: str = None):
    if file_name is None:
        file_name = 'sign_map.csv'
    df = pd.read_csv(os.path.join(root, file_name), sep=';')

    c_to_u_map = {}

    for _, row in df.iterrows():
        c_to_u_map[row['most_common']] = row['unicode']

    return c_to_u_map

def init_logging(exp_settings):
    """
    Create log dir with exp_id if it does not exist and run
    :param exp_settings:
    :return:
    """
    timestamp = time.strftime('%m%d_%H%M%S')
    dir_name = timestamp
    full_dir = os.path.join('output', 'results', str(exp_settings.name), dir_name)
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(os.path.join('output', 'results', str(exp_settings.name), dir_name, 'models'), exist_ok=True)

    # This creates an indication that the current experiment is not finished (or crashed).
    with open(os.path.join(full_dir, 'running.txt'), 'w'):
        pass

    return full_dir

def get_image_statistics(img_type = 'SketchB', log=False):
    if isinstance(img_type, list):
        img_type = img_type[0]

    if 'sketch' in img_type.lower():
        if log:
            print('Using sketch stats')
        mean, std = [0.7495, 0.7495, 0.7495], [0.0983, 0.0983, 0.0983]   # Calculated on SketchB
    elif 'normal' in img_type.lower():
        if log:
            print("Using normal stats")
        mean, std = [0.4995, 0.5195, 0.9353], [0.0244, 0.0287 ,0.0038]  # Calculated on Normal
    elif 'color' in img_type.lower():
        if log:
            print("Using color stats")
        # mean, std = [0.5470, 0.3306, 0.2130], [0.0436, 0.0185, 0.0099]   # Calculated on ColorA
        # mean, std = [0.6908, 0.4247, 0.2775], [0.0236, 0.0126, 0.0088]   # Calculated on Color00
        mean, std = [0.5391, 0.3292, 0.2154], [0.0449, 0.0186, 0.0102]     # Calculated on ColorA-H
    elif 'grayscale' in img_type.lower():
        if log:
            print("Using grayscale stats")
        mean, std = [0.3705, 0.3705, 0.3705], [0.0248, 0.0248, 0.0248]  # Calculated on v2.grayscale()
    else:
        raise ValueError(f"Unknown img type {img_type}")

    return mean, std


