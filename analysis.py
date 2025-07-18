import argparse
import json
import logging
import random
import re
from collections import defaultdict, Counter
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.stats
import torch
import torchvision.models
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib import font_manager
import os

from src import *

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams["font.family"] = "Yrsa"
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


def get_all_result_files(exp_id) -> Sequence[str]:
    """
    Find all predictions.csv from a particular exp_id. Return as a list of full_paths.
    """
    result_folder = os.path.join('./output/results', str(exp_id))
    all_results = []

    for result_id in os.listdir(result_folder):
        result_file = os.path.join(result_folder, result_id, 'predictions.csv')
        if os.path.exists(result_file):
            all_results.append(result_file)
    return all_results

def convert_old_names(df: pd.DataFrame, new_split_file = None, val=True):
    """
    Most of the results are made with old image names that were not consistent. They need to be converted
    to the new names, so they can work with updated dataset versions.

    All names that occur in the new_split_file are added as a row with nan's. This means we do not have
    to check for missing keys at a later level. This should probably not be passed when only concerning accuracies
    etc., but when using datasets (e.g. in TSNE plots) it is easy to have all keys that are in the dataset.
    """
    conversion = pd.read_csv('./output/results/old_to_new_name_conversion.csv', sep=';', index_col='old_name')
    img_type = df.iloc[0, 0].rsplit("_", maxsplit=1)[-1].split('.')[0]

    def convert_name(x):
        try:
            lookup = x.replace(img_type, 'TYPE')
            return conversion.loc[lookup]['new_name'].replace('TYPE', img_type)
        except KeyError:
            return x

    df['img_file'] = df['img_file'].apply(convert_name)

    if new_split_file is not None:
        split_file = pd.read_csv(new_split_file, sep=' ', header=None, names=['name', 'train', 'val', 'test'])
        split_file['name'] = split_file['name'].str.replace('TYPE', img_type)
        if val:
            split_file = split_file[split_file['val'] == 1]
        else:
            split_file = split_file[split_file['train'] == 1]

        missing = set(split_file['name']) - set(df['img_file'])
        for name in missing:  # TODO: row insertion is slow, but I don't need this often
            df.loc[len(df)] = [name, *[np.NAN for _ in range(df.shape[1] - 1)]]

    return df

def load_predictions(result_files: Sequence[str]) -> pd.DataFrame:
    """
    Load all prediction files into a single dataframe with the result_id as an additional column
    """
    all_dfs = []
    for rf in result_files:
        df = pd.read_csv(rf, sep=';')
        df['result_id'] = rf.split('/')[-2]
        all_dfs.append(df)

    print(f"{len(all_dfs)} experiments w lengths: {' - '.join([str(len(df)) for df in all_dfs])}")
    df = pd.concat(all_dfs)
    convert_old_names(df)
    return df

def add_topk_column(predictions: pd.DataFrame):
    """
    Adds column to predictions dataframe with values 'k', that indicate which k-th prediction was the correct one, 0
    stands for no correct prediction.
    :param predictions: data frame, predictions should be in order in columns with p[number] header.
    """
    pred_values = predictions.filter(regex=r'p\d+').to_numpy()
    pred_match = np.equal(pred_values, np.tile(predictions['target'].to_numpy(), (pred_values.shape[1], 1)).T)

    # Set pred_k to the i-th prediction that is correct. If none are correct, set to zero.
    pred_k = np.argmax(pred_match, axis=1) + 1
    pred_k[~np.any(pred_match, axis=1)] = 0

    predictions['pred_k'] = pred_k

def accuracy_from_topk(topk: np.ndarray, out_len=50):
    """
    Creates an array of accuracies based on topk array. This array has for every example the value k, meaning the
    k-th most likely prediction was the correct one. If this includes 0, it was never correct.
    :param topk: numpy array with topk values
    :param out_len: convenience, did not want to calculate what the max output value is
    :return: array with accuracies going from top-1 until top-[out_len].
    """
    values, counts = np.unique(topk, return_counts=True)
    counts = counts / len(topk)

    acc_arr = np.zeros(out_len)
    acc_arr[values] = counts
    # Don't include the first because 0 means it was not correct
    return np.cumsum(acc_arr[1:])

def join_predictions_with_meta(predictions: pd.DataFrame, meta_data_path: str = None, root: str = '../data'):
    """
    Add meta-data to predictions. It uses the data available in the sign_meta_data file, which is constructed when
    creating a dataset. It'll either look for it in the provided root dir, or uses a provided file at meta_data_path.

    The metadata frame is joined with the predictions using the name of the sign, minus the color information which
    is expected to be at the end, after the last '_'.

    :param predictions: predictions dataframe
    :param root: root path, used if not meta_data_path is provided. Should contain 'sign_meta_data.csv'.
    :param meta_data_path: path of the metadata
    :return: the predictions df joined with metadata
    """
    if meta_data_path is None:
        meta_data = pd.read_csv(f'{root}/sign_meta_data.csv', sep=';')
    else:
        meta_data = pd.read_csv(meta_data_path, sep=';')

    predictions['name'] = predictions['img_file'].apply(lambda x: x.rsplit('_', maxsplit=1)[0])
    predictions = predictions.set_index('name')

    meta_data['poly_area_sq'] = np.sqrt(meta_data['poly_area'])
    meta_data['bbox_area_sq'] = np.sqrt(meta_data['bbox_area'])
    meta_data['overlap_sq'] = np.sqrt(meta_data['overlap'])
    meta_data['provenience'] = meta_data['tablet'].str.extract(r'([A-Za-z]+)')

    join_df = predictions.join(meta_data.set_index('name'), rsuffix='_meta')
    return join_df

def filter_predictions(predictions: pd.DataFrame, include: Sequence[str], exclude: Sequence[str], cls_to_idx = None):

    if include is not None:
        for incl_rule in include:
            column, values = incl_rule.split(':')
            values = values.split('-')

            if column == 'target' and cls_to_idx is not None:
                values = [cls_to_idx[v] for v in values]

            predictions = predictions[predictions[column].isin(values)]

    if exclude is not None:
        for excl_rule in exclude:
            column, values = excl_rule.split(':')
            values = values.split('-')

            if column == 'target' and cls_to_idx is not None:
                values = [cls_to_idx[v] for v in values]

            predictions = predictions[~predictions[column].isin(values)]

    return predictions


def accuracy_from_pred(predictions: pd.DataFrame, root, class_avg=False, top_k=5, idx_to_class = None, print_result=True):
    add_topk_column(predictions)

    if class_avg:
        all_dicts = []
        class_counts = get_class_counts(root)
        groups = predictions.groupby(['target', 'result_id'])

        for name, g in groups:
            acc_arr = accuracy_from_topk(g['pred_k'].to_numpy())
            acc_dict = {'target': name[0], 'result_id': name[1]}
            acc_dict.update({f'top{i}': v for i, v in enumerate(acc_arr)})
            all_dicts.append(acc_dict)

        result_df = pd.DataFrame.from_records(all_dicts)
        all_results = {}

        class_wise_results = result_df.groupby('target')

        for cls, cr in class_wise_results:
            res = cr.filter(regex='top').agg(['mean', 'std'])
            mean, std = res.loc['mean'].to_numpy()[:top_k], res.loc['std'].to_numpy()[:top_k]

            cls_name = idx_to_class[cls] if idx_to_class is not None else cls
            all_results[cls_name] = [mean, std]
            # if print_result:
            #     num_val_samples = sum(predictions['target'] == cls)
            #     print(f"{cls_name};{num_val_samples};{';'.join([f'{m*100:.3f}' for m in mean])};"
            #           f"{';'.join([f'{s*100:.3f}' for s in std])}")

        avg_cls_results = result_df.groupby('result_id').mean()
        avg_cls_results = avg_cls_results.filter(regex='top').to_numpy()
        mean, std = avg_cls_results.mean(axis=0)[:top_k], avg_cls_results.std(axis=0)[:top_k]

        if print_result:
            for k, (m, s) in enumerate(zip(mean, std)):
                # print(f"Top {k+1}: {m*100:.2f}% (±{s*100:.2f})")
                print(f"Top {k+1};{m*100:.2f};{s*100:.2f}")

        return all_results

    else:
        if 'result_id' in predictions.columns:
            groups = predictions.groupby('result_id')
        else:
            groups = [('all', predictions)]

        accuracies = []
        for name, g in groups:
            accuracies.append(accuracy_from_topk(g['pred_k'].to_numpy())[:top_k])
        accuracies = np.array(accuracies)
        mean, std = accuracies.mean(axis=0), accuracies.std(axis=0)

        if print_result:
            for k, (m, s) in enumerate(zip(mean, std)):
                # print(f"Top {k+1}: {m*100:.2f}% (±{s*100:.2f})")
                print(f"Top {k+1};{m*100:.2f};{s*100:.2f}")

    return None

def accuracy_by_type(predictions, root = '../data', top_k=5, group_type: str | Sequence[str] = 'visual_face', print_result=True,
                     plot=False, plot_k=1, bins=10):
    add_topk_column(predictions)
    result_groups = predictions.groupby('result_id')
    results = []

    for r_name, result_df in result_groups:
        t_types = group_type if isinstance(group_type, list) else [group_type]
        for gt in t_types:
            group_col_type = result_df.dtypes[gt].kind
            if group_col_type == 'f':
                # result_df[gt] = pd.cut(result_df[gt], bins=bins)
                result_df[gt] = pd.qcut(result_df[gt], q=bins, duplicates='drop')

        groups = result_df.groupby(group_type, observed=True)

        for name, group in groups:
            acc_arr = accuracy_from_topk(group['pred_k'].to_numpy())
            result_dict = {'name': name, 'size': len(group), 'r_name': r_name}
            result_dict.update({f'top{i}': v for i, v in enumerate(acc_arr)})
            results.append(result_dict)

    results = pd.DataFrame.from_records(results)
    avg_values = results.pivot_table(values=[f'top{i}' for i in range(top_k)], index='r_name').to_numpy()
    results = results.groupby('name')

    results_values = {}
    for name, row in results:
        row_acc = row.filter(regex='top').to_numpy()[:, :top_k]
        mean, std = row_acc.mean(axis=0), row_acc.std(axis=0)
        results_values[name] = [mean, std, row.iloc[0, 1]]

        if print_result:
            print(f"{';'.join([str(n) for n in name])};{int(row.iloc[0, 1])};{';'.join([f'{m * 100:.3f}' for m in mean])};;"
                  f"{';'.join([f'{s * 100:.3f}' for s in std])}")

    avg_mean, avg_std = np.mean(avg_values, axis=0), np.std(avg_values, axis=0)
    total_values = sum(v[2] for v in results_values.values())
    # results_values[('zMean', )] = [avg_mean, avg_std, total_values)]
    print(f"Mean;{total_values};{';'.join([f'{m * 100:.3f}' for m in avg_mean])};{';'.join([f'{s * 100:.3f}' for s in std])}")

    if plot:
        fig, axes = plt.subplots(1, 1, figsize=(6.5, 3.5))
        names, mean_values, std_values, num_s = zip(*[(key, results_values[key][0][plot_k-1],
                                                       results_values[key][1][plot_k-1], results_values[key][2])
                                                for key in sorted(results_values, reverse=True)])
        y_ticks = np.arange(len(mean_values))
        axes.barh(y_ticks, mean_values, xerr=std_values, capsize=5, color='#9939c4', zorder=10)
        for y, v in zip(y_ticks, mean_values):
            axes.text(0.03, y, f"{v*100:.1f}%", color='white', fontweight='bold', ha='left', va='center', zorder=15)

        axes.set_yticks(y_ticks)
        names = [' '.join([str(n) for n in tick_name]) for tick_name in names]
        axes.set_yticklabels(f"{n} ({s})" for n, s in zip(names, num_s))
        x_ticks = np.arange(0, 1.01, 0.2)
        axes.set_xticks(x_ticks)
        axes.set_xticklabels([f"{xt*100:.0f}%" for xt in x_ticks])

        axes.set_xlim(0, 1.01)
        axes.grid(axis='x', zorder=0)
        for spine in ['bottom', 'top', 'left', 'right']:
            axes.spines[spine].set_visible(False)

    return results_values

def accuracy_by_type_vs_class(predictions, root, top_k, group_type, dataset: CuneiformClassification, **kwargs):
    results = accuracy_by_type(predictions, root, top_k, [*group_type, 'target'], print_result=False, **kwargs)

    group_names = sorted(list(set(k[:-1] for k in results.keys())))
    classes = range(max(dataset.labels))

    for c in classes:
        g_results = []
        for gn in group_names:
            try:
                g_results.append(results[*gn, c])
                # TODO: below is not tested, but python 3.10 seems to complain about the above
                # gn_c = gn.extend(c)
                # g_results.append(results[gn_c])
            except KeyError:
                g_results.append([[0], [0], 0])
        print(f"{dataset.idx_to_class[c]};{';'.join([str(m[0][top_k - 1]) for m in g_results])};"
              f"{';'.join([str(m[2]) for m in g_results])}")

    print(group_names)

def list_mistakes_cross(predictions: pd.DataFrame, idx_to_class = None, min_count=0, symmetric=False):
    num_results = len(predictions['result_id'].unique())
    mistakes = predictions[~predictions['p1'].eq(predictions['target'])]
    mistake_count = defaultdict(lambda: 0)

    for _, row in mistakes.iterrows():
        mis_key = (row['target'], row['p1'])
        if symmetric:
            mis_key = tuple(sorted(mis_key))
        mistake_count[mis_key] += 1

    mistake_count = sorted([(k, v) for k, v in mistake_count.items()], key=lambda x: x[1])

    for m in mistake_count:
        if idx_to_class is not None:
            target = idx_to_class[m[0][0]]
            pred = idx_to_class[m[0][1]]
        else:
            target, pred = m[0]

        # Ratios don't need to be rescaled because both numbers are rescaled
        cls_ratio = m[1] / (predictions['target'].eq(m[0][0])).sum()
        global_ratio = m[1] / len(predictions)
        num_mistakes = m[1] / num_results
        if num_mistakes >= min_count:
            print(f"{target};{pred};{num_mistakes};{cls_ratio*100:.3f};{global_ratio*100:.3f}")

def list_mistakes_by_sign(predictions, idx_to_class=None):
    num_results = len(predictions['result_id'].unique())
    mistakes = predictions[~predictions['p1'].eq(predictions['target'])]
    mistake_count = defaultdict(lambda: defaultdict(lambda: 0))

    for _, row in mistakes.iterrows():
        mistake_count[row['target']][row['p1']] += 1

    for sign, counts in mistake_count.items():
        num_mistakes = sum(counts.values()) / num_results
        num_signs = (predictions['target'].eq(sign)).sum() / num_results

        counts = sorted([(w, v / num_results) for w, v in counts.items()], key=lambda x: x[1], reverse=True)
        sign_mistakes_repr = ', '.join([f'{idx_to_class[w]} ({v})' for w, v in counts])

        print(f"{idx_to_class[sign]};{num_mistakes};{num_signs};{num_mistakes/num_signs*100:.3f};"
              f"{len(counts)};{sign_mistakes_repr}")

def visualize_mistakes(pred: pd.DataFrame, root, type='normal', images=None, idx_to_class=None, max_vis=9):
    """
    Visualize
        * images directly based on given string
        * images that match target:pred
        * images that match target

    *:* -> show just any prediction
    *:x -> show any mistake larger than top 5
    *:+ -> show top-1 mistakes that is corrected in 2, 3, 4, 5
    *:[NUM] -> show examples that are exactly top-NUM correct
    """

    pred = join_predictions_with_meta(pred)
    add_topk_column(pred)

    if re.match(r'.+:.+', type):
        class_to_idx = {v: k for k, v in idx_to_class.items()}
        target, p1 = type.split(':')
        if target == '*':
            if p1 == 'x':
                matched_mistakes = pred[~(pred['pred_k'].isin([1, 2, 3, 4, 5]))]
            elif p1 == '+':
                matched_mistakes = pred[(pred['pred_k'].isin([2, 3, 4, 5]))]
            elif p1 == '*':
                matched_mistakes = pred
            else:
                matched_mistakes = pred[(pred['pred_k'].isin([int(p1)]))]
        elif p1 == '*':
            target_idx = class_to_idx[target]
            matched_mistakes = pred[pred['target'].eq(target_idx)]
        elif p1 == 'x':
            target_idx = class_to_idx[target]
            matched_mistakes = pred[pred['target'].eq(target_idx) & ~(pred['p1'].eq(target_idx))]
        elif p1 == '5':
            target_idx = class_to_idx[target]
            matched_mistakes = pred[pred['target'].eq(target_idx) & ~(pred['pred_k'].isin([1, 2, 3, 4, 5]))]
        else:
            target_idx, pred_idx = class_to_idx[target], class_to_idx[p1]
            print(target_idx, pred_idx)
            matched_mistakes = pred[pred['target'].eq(target_idx) & pred['p1'].eq(pred_idx)]

        if len(matched_mistakes) > max_vis:
            images = random.sample(matched_mistakes['img_file'].to_list(), max_vis)
        else:
            images = matched_mistakes['img_file'].to_list()

    elif re.match(r'f_[a-z]+_[a-z]+', type):  # should go before previous as this matches that too.
        visual_face, annotated_face = type.split('_')[1:]
        print(visual_face, annotated_face)
        matched_mistakes = pred[(pred['visual_face'] == visual_face) & (pred['annotated_face'] == annotated_face) &
                                (pred['p1'] != pred['target'])]

        images = matched_mistakes['img_file'].to_list()
        random.shuffle(images)
        images = images[:max_vis]

    elif re.match(r'f_[a-z]+', type):
        visual_face = type.split('_')[1]
        matched_mistakes = pred[(pred['visual_face'] == visual_face) & (pred['p1'] != pred['target'])]

        images = matched_mistakes['img_file'].to_list()
        random.shuffle(images)
        images = images[:max_vis]

    elif re.match(r's_[.\d]+', type):
        max_size = float(type.split('_')[1])
        print(max_size)
        print(pred['poly_area_sq'])
        # matched_mistakes = pred[(pred['poly_area_sq'] < max_size) & (pred['p1'] != pred['target'])]
        matched_mistakes = pred[(pred['poly_area_sq'] < max_size) & (~pred['pred_k'].isin([1, 2, 3, 4, 5]))]
        print(len(matched_mistakes))
        images = random.sample(matched_mistakes['img_file'].to_list(), max_vis)

    # Actual visualization
    if isinstance(images, str):
        fig, axes = plt.subplots(1, 2, subplot_kw=dict(box_aspect=1))
        img_info = pred[pred['img_file'].eq(images)]
        visualize_single_pred(img_info, images, root, idx_to_class, axes=axes)

        corr = img_info['target'].values[0]
        cls_name_to_unicode = load_cls_name_to_unicode_map(root='../data')
        fig.text(0.49, 0.9, ascii_to_unicode_atf(idx_to_class[corr]), ha="right", size=15, fontweight='bold',
                 va='center')
        fig.text(0.51, 0.9, cls_name_to_unicode[idx_to_class[corr]], ha='left', va='center', size=20,
                 fontproperties=get_cuneiform_font_prop())

    elif isinstance(images, list):
        print(images)
        num_rows = 2
        rows = len(images) if len(images) <= num_rows else num_rows
        cols = 2 * (1 if len(images) <= num_rows else 1 + (len(images) - 1) // num_rows)

        fig, axes = plt.subplots(rows, cols, subplot_kw=dict(box_aspect=1), figsize=(5 * cols / 2, 2.5 * num_rows),
                                 squeeze=False)

        # fig, axes = plt.subplots(rows, cols, subplot_kw=dict(box_aspect=1), figsize=(6.26, 1.5),
        #                          squeeze=False)

        for i, img in enumerate(images):
            # TODO: maybe weird place to reduce to single image here if multiple exist
            img_info = pred[pred['img_file'].eq(img)].iloc[0]
            row, col = i % num_rows, i // num_rows * 2
            visualize_single_pred(img_info, img, root, idx_to_class, [axes[row, col], axes[row, col + 1]])

def plot_accuracy_vs(pred: pd.DataFrame, root, k=5, vs='class_count', idx_to_class=None):
    import seaborn as sns

    acc = accuracy_from_pred(pred, root, top_k=25, class_avg=True, idx_to_class=idx_to_class, print_result=False)
    fig, axes = plt.subplots(1, 1, figsize=(4, 2.5))

    if vs == 'class_count':
        class_counts = get_class_counts(root)
        data_points = {key: (class_counts[key], acc[key][0][k-1]) for key in acc}
    else:
        raise ValueError(f"VS not implemented {vs}")

    x, y = zip(*list(data_points.values()))
    df = pd.DataFrame({'class_count': x, 'accuracy': y})
    # df['class_count'] = pd.qcut(df['class_count'], q=10)
    df['class_count'] = pd.cut(df['class_count'], bins=[20, 40, 60, 100, 250, 10_000], include_lowest=True)

    sns.despine(fig)
    axes.grid()

    sns.stripplot(
        data=df, x="class_count", y="accuracy",
        dodge=True, alpha=.2, legend=False)

    print(df.groupby('class_count')['accuracy'].mean())

    sns.pointplot(
        data=df, x="class_count", y="accuracy", linestyle="none", errorbar=None,
        marker="_", markersize=20, markeredgewidth=3,)

    xtick_labels = [f"{i.left:.0f} - {i.right:.0f}" for i in df['class_count'].cat.categories.to_list()]
    axes.set_xticklabels(xtick_labels)

    y_ticks = np.arange(0, 1.01, 0.10)
    axes.set_yticks(y_ticks)
    axes.set_yticklabels([f"{yt:.0f}" for yt in 100*y_ticks])

    axes.set_ylim(0, 1.0)


    # axes.scatter(x, y, zorder=10, color='#7a2d9a', edgecolors='k', marker="o", s=20)
    #
    # axes.set_xscale('log')
    # axes.set_xlim(10, max(x) * 1.1)
    # axes.set_ylim(-0.02, 1.02)
    #
    # axes.grid(zorder=0, which='both')
    # axes.set_xlabel("Num signs")
    # axes.set_ylabel(f"Top-{k} accuracy (%)")
    #
    # for spine in ['bottom', 'top', 'left', 'right']:
    #     axes.spines[spine].set_visible(False)

def plot_mistakes_loc(predictions, root, rel=True, bins=5):
    import seaborn as sns

    rv = accuracy_by_type(predictions, root, top_k=5, group_type=['cy', 'cx'], print_result=False, bins=bins)
    # rv = accuracy_by_type(predictions, root, top_k=1, group_type=['cy', 'cx', 'target'], print_result=False, bins=bins)

    fig, axes = plt.subplots(1, 1, figsize=(2, 1.5), subplot_kw=dict(box_aspect=1))
    acc_dic = defaultdict(float)
    class_count_dic = defaultdict(int)
    acc = []

    for key in rv:
        cx = key[0].left
        cy = key[1].left
        acc.append((rv[key][0][0] * 100))
        # rv: key, mean/std/count, topk
        acc_dic[(cx, cy)] += rv[key][0][0] * 100
        class_count_dic[(cx, cy)] += 1

    # acc = [acc_dic[k] / class_count_dic[k] for k in acc_dic]

    acc = np.array(acc)
    acc = acc.reshape(bins, bins)

    if rel:
        if bins == 5:
            acc -= np.array([  # Accuracy of ColorOO
                [78, 74, 71, 74, 67],
                [82, 74, 73, 70, 69],
                [82, 78, 74, 72, 71],
                [82, 76, 74, 72, 67],
                [79, 72, 65, 70, 63],
            ])

        elif bins == 3:
            acc -= np.array([
               [78, 73, 71],
               [81, 76, 71],
               [76, 70, 70],
            ])

    if rel:
        sns.heatmap(acc, annot=True, fmt='.1f', cmap='RdYlGn', cbar=False, vmin=-2, vmax=5, ax=axes)
    else:
        sns.heatmap(acc, annot=True, fmt='.1f', cmap='RdYlGn', cbar=False)

    axes.set_xticks([])
    axes.set_yticks([])

def wrong_correct_vs_normal(predictions):
    """
    Plot correct or not
    :param predictions:
    :return:
    """
    import seaborn as sns

    add_topk_column(predictions)
    predictions['correct'] = predictions['pred_k'] == 1

    normals = torch.tensor(predictions[['nx', 'ny', 'nz']].to_numpy())

    # light_dir = torch.tensor([[0, -1, -1]])          # ColorA
    # light_dir = torch.tensor([[-1, 0, -1]])          # ColorC
    # light_dir = torch.tensor([[0, 1, -1]])           # ColorE
    light_dir = torch.tensor([[1, 0, -1]])           # ColorG

    align = torch.cosine_similarity(light_dir, normals)
    predictions['light_align'] = align

    correct_kurt = predictions['light_align'][predictions['correct']].std()
    wrong_kurt = predictions['light_align'][~predictions['correct']].std()

    print(correct_kurt, wrong_kurt)


    # sns.violinplot(predictions, x='correct', y='light_align', cut=0)
    # plt.grid()

    sns.histplot(
        predictions, x="light_align", hue="correct",
        multiple="layer", fill=False, element='poly', stat='probability', linewidth=2.5)

    non_nan_idx = predictions['light_align'].notnull()
    print(scipy.stats.pointbiserialr(predictions['correct'][non_nan_idx], predictions['light_align'][non_nan_idx]))

    # sns.histplot(
    #     predictions, x="light_align", hue="correct",
    #     multiple="stack", fill=True, element='bars', stat='probability', linewidth=1)

    # correct_nz = predictions['nz'][predictions['correct'] & predictions['nz'].notnull()]
    # wrong_nz = predictions['nz'][~predictions['correct'] & predictions['nz'].notnull()]
    #
    # plt.boxplot([correct_nz, wrong_nz])




def main_analysis():
    """
    This looks for all predictions.csv files in an experiment folder and aggregates them and does a bunch of different
    analysis on them.
    :return:
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('exp_id', help='The ID of the experiment to use for this analysis')
    parser.add_argument('--type', default='acc', choices=['acc', 'mis', 'vis', 'acct', 'other'])
    parser.add_argument('--spec', nargs='*')
    parser.add_argument('-k', type=int, help='in some places this defines the top-k level', default=1)
    parser.add_argument('--include', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    args = parser.parse_args()

    exp_id = args.exp_id

    root = '../data_vat_o'
    # root = '../data_normals'
    split_file = 'train_val_split_cunes_hs_vat_o.txt'

    # The only reason that it is important to take the correct one is for the cls_to_idx maps
    # root = '../data_test'
    # split_file = 'train_val_split_08_02_0411_1142.txt'

    all_results = get_all_result_files(exp_id)
    if args.type == 'vis':  # If visualizing the results, we only want a single model
        all_results = all_results[0:1]

    all_results = ['./56_nlp_results.csv']
    # all_results = all_results[0:1]
    # print(all_results)

    # The only reason we're using this is to get the idx_to_class, but with lazy_load it is probably fine
    val_set = CuneiformClassification(root, min_nb_img=20, split='val', transform=None, lazy_load=True, img_type="SketchB",
                                      split_file=split_file)

    predictions = load_predictions(all_results)
    predictions = join_predictions_with_meta(predictions, root=root)
    predictions = filter_predictions(predictions, args.include, args.exclude, val_set.class_to_idx)

    if args.type == 'acc':
        print("Average")
        accuracy_from_pred(predictions, root, top_k=args.k, class_avg=False, idx_to_class=val_set.idx_to_class)
        print("Class wise")
        accuracy_from_pred(predictions, root, top_k=args.k, class_avg=True, idx_to_class=val_set.idx_to_class)

    elif args.type == 'mis':
        list_mistakes_cross(predictions, idx_to_class=val_set.idx_to_class, min_count=0, symmetric=False)
        # list_mistakes_by_sign(predictions, idx_to_class=val_set.idx_to_class)

    elif args.type == 'acct':
        accuracy_by_type(predictions, root, top_k=args.k, group_type=args.spec, plot_k=args.k, plot=True, bins=30)
        # accuracy_by_type_vs_class(predictions, root, top_k=args.k, group_type=args.spec, dataset=val_set, bins=3)

        plt.title(exp_id)
        plt.tight_layout()
        # plt.savefig(f"./graphs/acc_by_type/acc_by_type_k=1_type={args.spec}_exp_id={exp_id}.svg", dpi=300)
        plt.show()

    elif args.type == 'vis':
        visualize_mistakes(predictions, root, type=args.spec[0], idx_to_class=val_set.idx_to_class,
                           max_vis=8)
        plt.tight_layout()
        # plt.savefig(f"./graphs/examples/na_3.svg")
        plt.show()

    else:
        # plot_mistakes_loc(predictions, root, rel=False, bins=3)
        # plot_accuracy_vs(predictions, root, k=args.k, vs='class_count', idx_to_class=val_set.idx_to_class)

        # wrong_correct_vs_normal(predictions)


        plt.tight_layout()
        plt.savefig(f'./graphs/area/area_rel_{exp_id}.svg')
        # plt.savefig(f'./graphs/top_{args.k}_vs_class_count_new.svg', dpi=300)
        plt.show()





if __name__ == '__main__':
    main_analysis()
