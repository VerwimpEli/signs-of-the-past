"""
Needs to be refactored, for now just a place to dump the relevant methods.
"""
import json
import os
from collections import defaultdict, Counter

from .utils import unicode_to_ascii_atf
from .data.utils import get_target


def load_full_sign_map(root, file_name=None):
    if file_name is None:
        file_name = f'{root}/sign_2_UnicodeCuneiform_map.txt'

    sign_map = {}

    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            try:
                ascii_atf = unicode_to_ascii_atf(line[0])
                ascii_atf = ascii_atf.replace('/', '-')  # Replace all slashes with dashes (/ is annoying)
                unicode_nb = line[1]
            except IndexError:
                # some weird signs had mistakes, but not important here
                continue
            sign_map[ascii_atf] = unicode_nb

    # some conversions that aren't in the root file
    sign_map.update({
        '1(gur)': '1(asz)',
        '2(gur)': '2(asz)',
        'eri4': 'uru'
    })

    return sign_map


def create_reduced_sign_map():
    root = '../data_vat_o'
    sign_map = load_full_sign_map(root)
    reverse_sign_map = defaultdict(list)

    annot_dir = f'{root}/annotations'
    annot_files = os.listdir(annot_dir)

    target_counts = defaultdict(int)

    missing = defaultdict(list)

    for annot in annot_files:
        tablet_name, face = annot.rsplit('_', maxsplit=1)
        face = face.split('.')[0]

        annot_file = f'{root}/annotations/{tablet_name}_{face}.json'

        with open(annot_file, 'r') as f:
            data = json.load(f)

        for i, key in enumerate(data):
            target = get_target(data[key])
            try:
                unicode_target = sign_map[target]
                reverse_sign_map[unicode_target].append(target)
                target_counts[unicode_target] += 1
            except KeyError as e:
                if '#' in target or '*' in target or '?' in target:
                    continue
                # missing[target].append([annot, key])

    # missing = sorted([(k, v) for k, v in missing.items()], key=lambda x: len(x[1]), reverse=True)
    # for sign in missing:
    #     print(sign[0])
    #     for line in sign[1]:
    #         print(line)
    #     print()

    target_counts = sorted([(t, v) for t, v in target_counts.items()], key=lambda x: x[1], reverse=True)
    for t, v in target_counts:
        counts = Counter(reverse_sign_map[t])
        counts = counts.most_common()

        count_str = ''
        for elem in counts:
            count_str += f'{elem[0]} ({elem[1]}), '
        count_str = count_str[:-2]

        print(f"{t};{v};{' '.join(set(reverse_sign_map[t]))};{count_str};"
              f"{counts[0][0]};")

    print(f"All signs: {len(target_counts)} | Above 20: {len([1 for tc in target_counts if tc[1] >= 20])}"
          f" | Total sign count {sum(tc[1] for tc in target_counts)}")


if __name__ == '__main__':
    create_reduced_sign_map()
