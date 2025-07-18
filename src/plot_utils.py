import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .utils import get_cuneiform_font_prop, load_cls_name_to_unicode_map, ascii_to_unicode_atf


def draw_bounding_box(ax, bbox, color='red', label=None, label_pos='top'):
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ax.add_patch(Rectangle((bbox[0], bbox[1]), bw, bh, fill=False, lw=2, edgecolor=color))
    ax.add_patch(Rectangle((bbox[0], bbox[1]), bw, bh, fill=True, lw=0, edgecolor=None,
                           facecolor=color, alpha=0.1))


    if label is not None:
        if label_pos == 'top':
            x, y = bbox[0], bbox[1]
            va = 'bottom'
        else:
            x, y = bbox[0], bbox[3]
            va = 'top'

        if color in ['green', 'blue', 'purple']:
            text_color = 'white'
        else:
            text_color = 'black'

        ax.text(x, y, label, color=text_color, backgroundcolor=color, va=va, ha='left',
                fontsize=10)


def visualize_single_pred(img_info=None, img_file=None, root=None, idx_to_class=None, axes=None, img=None,
                          pred=None, conf=None, color='#b30024'):
    from PIL import Image

    if axes is None:
        fig, axes = plt.subplots(1, 2, subplot_kw=dict(box_aspect=1))

    if img is None:
        img = Image.open(os.path.join(root, 'sign_crops', img_file))

    axes[0].imshow(img)
    axes[0].axis('off')

    if img_info is not None:
        topk = 6
        corr = img_info['target']
        pred = img_info[[f'p{i+1}' for i in range(topk)]][::-1]
        conf = img_info[[f'c{i+1}' for i in range(topk)]][::-1]
    else:
        topk = len(pred)
        corr = None

    colors = np.array([color] * topk)
    colors[corr == pred] = '#04b36f'

    axes[1].barh(np.arange(len(conf)), conf, zorder=10, color=colors)
    setup_bar_axes(axes[1])

    font_prop = get_cuneiform_font_prop()
    cls_name_to_unicode = load_cls_name_to_unicode_map(root)

    # Add text to bars
    for i, (p, c), in enumerate(zip(pred, conf)):
        if c > 0.0005:
            out = c < 0.4
            text_x = c + 0.05 if out else 0.05
            color = 'k' if out else 'white'
            sign_str = ascii_to_unicode_atf(idx_to_class[p])

            t = axes[1].text(text_x, i, sign_str, va='center', color=color, zorder=20,
                         fontweight='bold', size=12)

            # Some deep plt sorcery
            r = axes[1].get_figure().canvas.get_renderer()
            width = t.get_window_extent(renderer=r).transformed(axes[1].transData.inverted()).width

            axes[1].text(text_x + width, i, cls_name_to_unicode[idx_to_class[p]], va='center', color=color, zorder=20,
                         fontweight='bold', fontproperties=font_prop, size=20)

    # Add correct sign to image
    if corr is not None:
        corr_str = ascii_to_unicode_atf(idx_to_class[corr])
        axes[0].annotate(corr_str, xy=(1, 0), fontsize=12, xycoords='axes fraction', annotation_clip=True,
                         fontweight='bold', bbox=dict(facecolor='white', alpha=0.95, edgecolor='white'),
                         ha='right', va='bottom')


def setup_bar_axes(ax):
    ax.set_xlim((0, 1.05))
    ax.grid(axis='x', zorder=0)
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([f"{x * 100:.0f}%" for x in np.arange(0, 1.01, 0.2)])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params('both', length=0, width=1)