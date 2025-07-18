import argparse
import os
from typing import Sequence

import matplotlib.patches
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from torch.utils.data import DataLoader

# TODO: this shouldn't be here.
from train_class import get_transforms

from src import CuneiformClassification


@torch.no_grad()
def calc_features(model: torch.nn.Module, dataset: CuneiformClassification, device: str = 'cuda'):
    loader = DataLoader(dataset, batch_size=8, num_workers=4)

    model = model.to(device)
    model.eval()
    # Replace final layer with identity layer. Could create forward hooks to not destroy model, but that's more complicated
    # than it should be here.
    model.fc = torch.nn.Sequential()

    all_features, all_targets = [], []

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        features = model(data)
        all_features.append(features)
        all_targets.append(target)

    all_features = torch.cat(all_features)
    all_targets = torch.cat(all_targets)

    return all_features, all_targets

def filter_features(x_data: torch.Tensor, y_data: torch.Tensor, labels: Sequence[int]):
    idx = torch.zeros_like(y_data, dtype=torch.bool)
    for ci in labels:
        idx = idx | (y_data == ci)
    x_data, y_data = x_data[idx], y_data[idx]
    img_idx = torch.argwhere(idx).flatten()
    return x_data, y_data, img_idx

def calc_class_means(x_data, y_data: torch.Tensor):
    unique_y = y_data.unique(sorted=True)
    cls_means = []

    for target in unique_y:
        cls_data = x_data[y_data == target]
        cls_means.append(cls_data.mean(dim=0))

    return torch.stack(cls_means), unique_y

def plot_class_mean_similarity(cm_x, cm_y, idx_to_class = None):
    import seaborn as sns

    for n in range(20):
        print(n)
        n_clusters = 193 - n
        kmeans = KMeans(n_clusters=n_clusters).fit(cm_x)
        order = np.argsort(kmeans.labels_)

        cm_x_ord = cm_x[order]
        sim = torch.zeros(len(cm_y), len(cm_y))

        for i in range(len(cm_y)):
            sim[i] = torch.pairwise_distance(cm_x_ord, cm_x_ord[i])
            # sim[i] = 1.0 - torch.cosine_similarity(cm_x, cm_x[i])
            # sim[i, :i+1] = np.nan  # Only keep upper triangular matrix

        most_sim_value, most_sim_idx = torch.topk(sim.flatten(), k=1000, largest=False)
        most_sim_idx = np.array(np.unravel_index(most_sim_idx.numpy(), sim.shape)).T

        for k in range(n_clusters):
            cluster_y = cm_y[kmeans.labels_ == k]
            cluster_names = [idx_to_class[y.item()] for y in cluster_y]
            if len(cluster_names) > 1:
                print(f"Cluster {k}: {' - '.join(cluster_names)}")


    # for row, value in zip(most_sim_idx, most_sim_value):
    #     # print(f"{idx_to_class[row[0]]:10s} & {idx_to_class[row[1]]:10s} : {value:.3f}")
    #     print(f"{idx_to_class[order[row[0]]]};{idx_to_class[order[row[1]]]};{value:.3f}")

    # sim[sim > 12.5] = np.nan
    # tick_labels = [idx_to_class[i.item()] for i in cm_y[order]]
    # sns.heatmap(sim, annot=False, fmt='.2f', xticklabels=tick_labels, yticklabels=tick_labels,
    #             cmap='Oranges_r', vmin=5, vmax=25)

def knn_classification(x_data, y_data: torch.Tensor, x_data_val, y_data_val, ds: CuneiformClassification):
    """
    Perform classification using
    :param x_data:
    :param y_data:
    :param x_data_val:
    :param y_data_val:
    :return:
    """
    scaler = MinMaxScaler()
    x_data = scaler.fit_transform(x_data)
    x_data_val = scaler.transform(x_data_val)

    # dim_red = PCA(n_components=50)
    # x_data = dim_red.fit_transform(x_data)
    # x_data_val = dim_red.transform(x_data_val)

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(x_data, y_data)


    # Top-k KNN accuracy
    # _, pred_neighbors = knn.kneighbors(x_data_val)
    # pred_neighbors = y_data[pred_neighbors]
    #
    # knn_topk_pred = []
    # for row, target in zip(pred_neighbors, y_data_val):
    #     pred, counts = torch.unique(row, return_counts=True)
    #     pred_sorted = pred[torch.argsort(counts, descending=True)]
    #     top_k = torch.eq(pred_sorted, target)
    #     top_k = torch.argwhere(top_k)[0].item() + 1 if torch.any(top_k) else 0
    #     knn_topk_pred.append(top_k)
    # knn_topk_pred = np.array(knn_topk_pred)
    # print(accuracy_from_topk(knn_topk_pred))

    knn_pred = knn.predict(x_data_val)
    knn_pred = torch.tensor(knn_pred)

    unique_labels = y_data_val.unique()
    class_acc = []
    for ul in unique_labels:
        label_idx = torch.argwhere(y_data_val.eq(ul))
        label_acc = torch.eq(knn_pred[label_idx], ul)
        class_acc.append(label_acc.sum().item() / len(label_idx))
        print(f"{ds.idx_to_class[ul.item()]};{class_acc[-1]}")

    acc = sum(torch.eq(knn_pred, y_data_val)) / len(y_data_val)
    print(f"Class-accuracy: {np.mean(class_acc) * 100:.3f}")
    print(f"Top-1 accuracy: {acc * 100:.3f}")

def compute_gap_statistic(X, k_max, n_replicates=10):
    """
    Compute the Gap Statistic for a range of cluster numbers.

    Parameters:
    X: array-like, shape (n_samples, n_features)
        The input data.
    k_max: int
        The maximum number of clusters to evaluate.
    n_replicates: int
        The number of bootstrap samples.

    Returns:
    gap_values: list
        The calculated gap values for each k.
    """

    # Generate reference data from a uniform distribution
    def generate_reference_data(X):
        return np.random.uniform(low=X.min(dim=0).values, high=X.max(dim=0).values, size=X.shape)

    gap_values = []

    # Loop over a range of k values (1 to k_max)
    for k in range(1, k_max + 1):
        # Fit KMeans to the original data
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        original_inertia = kmeans.inertia_

        # Compute the average inertia for the reference datasets
        reference_inertia = []
        for _ in range(n_replicates):
            random_data = generate_reference_data(X)
            kmeans.fit(random_data)
            reference_inertia.append(kmeans.inertia_)

        # Calculate the Gap statistic
        gap = np.log(np.mean(reference_inertia)) - np.log(original_inertia)
        gap_values.append(gap)

    return gap_values

def compute_k_from_gap(x_data, std_lim = 1.75, plot=False):
    gap_values = compute_gap_statistic(x_data, 15, n_replicates=10)

    diff = np.diff(gap_values)
    diff_med, diff_std = np.mean(diff), np.std(diff)

    if plot:
        fig, axes = plt.subplots()
        axes.plot(gap_values, '-o')
        axes.plot(diff)
        axes.hlines([diff_med, diff_med + std_lim * diff_std], 0, 10, colors=['r', 'g'])
        axes.set_ylim((0.0, 2.0))
        plt.grid()

    print(np.nonzero(diff > (diff_med + diff_std))[0])
    k = np.nonzero(diff > (diff_med + std_lim * diff_std))[0].max() + 2
    return k

def find_sign_versions(x_data, y_data, labels, dataset, k: int = None, nb_examples: int = 3):
    x_data, y_data, img_idx = filter_features(x_data, y_data, labels)

    if k is None:
        k = compute_k_from_gap(x_data, plot=False)

    estimator = KMeans(k)
    estimator.fit(x_data)
    labels = torch.tensor(estimator.labels_)
    centroids = torch.tensor(estimator.cluster_centers_)

    fig, axes = plt.subplots(k, nb_examples, figsize=(10, k * 3.5))

    for i, lab in enumerate(labels.unique()):
        lab_idx = torch.nonzero(labels.eq(lab))
        dist_to_centroid = torch.pairwise_distance(x_data[lab_idx], centroids[lab]).flatten()
        closest_idx = dist_to_centroid.topk(k=nb_examples, largest=False).indices
        closest_idx = lab_idx[closest_idx]

        for j, idx in enumerate(closest_idx):
            img, _ = dataset[img_idx[idx]]
            axes[i, j].imshow(img)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

        axes[i, 0].set_ylabel(f"Version {i + 1}", fontsize=15)
        for j in range(nb_examples):
            axes[0, j].set_title(f"Example {j + 1}", fontsize=15)

    fig.tight_layout()

def mean_distance(x_data, y_data, dataset):
    meta_data = pd.read_csv(os.path.join(dataset.sign_root, '../sign_meta_data.csv'), sep=';')
    meta_data['provenience'] = meta_data['tablet'].str.extract(r'([A-Za-z]+)')
    meta_data = meta_data.set_index('name')

    all_proveniences = sorted(meta_data['provenience'].unique().tolist())

    meta_data['ds_idx'] = -1
    for i, img in enumerate(dataset.images):
        img = img.rsplit('_', maxsplit=1)[0]
        meta_data.loc[img, 'ds_idx'] = i

    target_groups = meta_data.groupby('target')
    results = []
    for t, tg in target_groups:
        if len(tg) < 100:
            continue

        target_result = {'target': t}
        means = {}

        for prov in all_proveniences:
            g = tg[tg['provenience'] == prov]

            if len(g) > 0:
                img_idx = [i for i in g['ds_idx'] if i > -1]
                means[prov] = torch.mean(x_data[img_idx], dim=0)
                target_result[prov] = len(img_idx)
            else:
                target_result[prov] = 0

        for i in range(len(all_proveniences)):
            for j in range(i+1, len(all_proveniences)):
                p1, p2 = all_proveniences[i], all_proveniences[j]
                try:
                    # target_result[f"{p2}-{p1}"] = torch.pairwise_distance(means[p1], means[p2]).item()
                    target_result[f"{p2}-{p1}"] = torch.cosine_similarity(means[p1], means[p2], dim=0).item()
                except KeyError:
                    target_result[f"{p2}-{p1}"] = np.NAN

        results.append(target_result)

    df = pd.DataFrame(results)
    print(df.to_csv(sep=';', index=False))


def tsne_plot(
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        dataset: CuneiformClassification = None,
        cls_to_show: Sequence[str] = None,
        x_data_val: torch.Tensor = None,
        y_data_val: torch.Tensor = None,
        val_dataset: CuneiformClassification = None,
        k: int = None,
        group_key: str = None
):
    """
    Create a tsne plot of the provided data. If cls_to_show is none, all data is shown. If not, only the data that
    corresponds to the provided class names is shown. The provided dataset is used to access to class_to_index map.
    If more than one sign is passed, the signs are colored according to their label. If only one is passed, k-means
    is applied and the clusters are colored accordingly. If k is None, the number of clusters is determined
    automatically using the GAP statistic. If k is passed, this is set as the number of clusters.

    :param x_data: the data to use in the TSNE plots. Typically, this is the training data.
    :param y_data: the labels corresponding to x_data.
    :param dataset: the dataset object from which x_data and y_data were created. This is used to access the images.
    :param cls_to_show: the names of classes to include.
    :param x_data_val: validation data used together with x_data in the plot.
    :param y_data_val: validation labels used vor x_data_val.
    :param val_dataset: the validation dataset from which x_data_val was created.
    :param k: number of clusters for k-means if only a single sign is shown.
    :param group_key: key in meta_data to color the samples. Currently only supports categorical columns (mostly provenience thus)
    """

    # Filter data based on the classes to show
    if cls_to_show is not None:
        cls_idx = [dataset.class_to_idx[cls] for cls in cls_to_show]
        x_data, y_data, img_idx = filter_features(x_data, y_data, cls_idx)
    else:
        img_idx = np.arange(len(y_data))

    # Include validation data in plot
    if x_data_val is not None:
        if cls_to_show is not None:
            cls_idx = [val_dataset.class_to_idx[cls] for cls in cls_to_show]
            x_data_val, y_data_val, img_idx_val = filter_features(x_data_val, y_data_val, cls_idx)
        else:
            img_idx_val = np.arange(len(x_data_val))

        # Include predictions
        # predictions = pd.read_csv('results/31/0327_201437/predictions.csv', sep=';')
        predictions = pd.read_csv('output/results/37/0508_144655/predictions.csv', sep=';')
        # predictions = analysis.convert_old_names(predictions, new_split_file='../data_test/train_val_split_08_02_0411_1142.txt')

        inferred_img_type = predictions.iloc[0, 0].rsplit("_", maxsplit=1)[-1].split('.')[0]
        if inferred_img_type != val_dataset.img_type[0]:
            predictions['img_file'] = predictions['img_file'].apply(
                lambda x: x.replace(inferred_img_type, val_dataset.img_type[0]))

        predictions = predictions.set_index('img_file')
        predictions = predictions.loc[val_dataset.images[img_idx_val]]

        correct_idx = np.nonzero(predictions['target'].eq(predictions['p1']).to_numpy())[0]
        wrong_idx = np.nonzero(~predictions['target'].eq(predictions['p1']).to_numpy())[0]

        tsne_data = torch.cat([x_data, x_data_val])
    else:
        tsne_data = x_data

    # If group key is not None, use metadata to color. Else if single class: compute k-means and color according to it.
    # If multiple classes, color classes.
    if group_key is not None:
        meta_data = pd.read_csv(os.path.join(dataset.sign_root, '../sign_meta_data.csv'), sep=';')
        meta_data['provenience'] = meta_data['tablet'].str.extract(r'([A-Za-z]+)')
        meta_data = meta_data.set_index('name')

        names = pd.Series(dataset.images[img_idx]).str.rsplit('_', n=1, expand=True)[0]
        cat_names = sorted(meta_data[group_key].unique().tolist())
        labels = pd.Categorical(meta_data.loc[names][group_key], categories=cat_names, ordered=True).codes

        val_names = pd.Series(val_dataset.images[img_idx_val]).str.rsplit('_', n=1, expand=True)[0]
        val_labels = pd.Categorical(meta_data.loc[val_names][group_key], categories=cat_names, ordered=True).codes

    elif cls_to_show is None or len(cls_to_show) > 1:
        values = sorted(y_data.unique().tolist())
        cat_names = [dataset.idx_to_class[v] for v in values]
        labels = pd.Categorical(y_data.tolist(), categories=values).codes
        if y_data_val is not None:
            val_labels = pd.Categorical(y_data_val.tolist(), categories=values).codes
    elif len(cls_to_show) == 1:
        if k is None:
            k = compute_k_from_gap(x_data, plot=False)
        estimator = KMeans(k, n_init=10)
        estimator.fit(x_data)
        labels = torch.tensor(estimator.labels_)
        cat_names = [f"Cluster {c+1}" for c in range(k)]
        if x_data_val is not None:
            val_labels = torch.tensor(estimator.predict(x_data_val))
    else:
       raise ValueError("Huh")


    # Calculate TSNE projections
    tsne = TSNE(n_components=2, perplexity=10, random_state=1997)
    projections = tsne.fit_transform(tsne_data)

    # Create scatter plots. Needs to be three different plots, a single plot cannot have different markers
    colors = np.array(['red', 'green', 'blue', 'orange', 'olive', 'cyan', 'pink', 'black', 'violet', 'crimson'])

    # fig, axes = plt.subplots(1, 1, figsize=(12.5, 7.5), subplot_kw=dict(box_aspect=1))
    fig, axes = plt.subplots(1, 1, figsize=(4.0, 4.0), subplot_kw=dict(box_aspect=1))
    alpha = 0.5

    # Scatter plot of training samples
    sc = axes.scatter(projections[:len(y_data), 0], projections[:len(y_data), 1], c=colors[labels], zorder=10, alpha=alpha,
                      linewidths=0)

    if x_data_val is not None:
        # Scatter plot of correct predictions
        sc_val_c = axes.scatter(projections[correct_idx + len(y_data), 0], projections[correct_idx + len(y_data), 1],
                                c=colors[val_labels[correct_idx]], zorder=10, marker='^', s=45, edgecolors='k')

        # sc_val_c = axes.scatter(projections[correct_idx + len(y_data), 0], projections[correct_idx + len(y_data), 1],
        #                         c=colors[val_labels[correct_idx]], zorder=10, alpha=alpha, linewidths=0)

        # Scatter plot of wrong predictions
        sc_val_w = axes.scatter(projections[wrong_idx  + len(y_data), 0], projections[wrong_idx  + len(y_data), 1],
                                c=colors[val_labels[wrong_idx]], zorder=10, marker='X', s=45, edgecolors='k')

        # sc_val_w = axes.scatter(projections[wrong_idx  + len(y_data), 0], projections[wrong_idx  + len(y_data), 1],
        #                         c=colors[val_labels[wrong_idx]], zorder=10, alpha=alpha, linewidths=0)

    # Make annotation that shows the label
    annot = axes.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                         bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"), zorder=15)
    annot.set_visible(False)

    # Add image annotation that shows the image over which is hovered
    img_box = OffsetImage(np.empty((200, 200)))
    img_box.image.axes = axes
    ab = AnnotationBbox(img_box, (0, 0), xybox=(150, -150), xycoords='data', boxcoords='offset points',
                        pad=0.5, arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=3"),
                        zorder=15)
    axes.add_artist(ab)
    ab.set_visible(False)

    # Update the annotation that is hovered above.
    prev_ind = -1  # Keep track of last ind that was shown and only print if something changes.
    def update_annot(scatter, ind, val=False, correct=True):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos

        if len(ind['ind']) == 1:
            nonlocal prev_ind
            s_idx = ind['ind'][0]
            if not val:
                print(dataset.images[img_idx[s_idx]])
                img, target = dataset[img_idx[s_idx]]
            else:
                c_img_idx = img_idx_val[correct_idx[s_idx]] if correct else img_idx_val[wrong_idx[s_idx]]
                img, target = val_dataset[c_img_idx]
                print(val_dataset.images[c_img_idx])

                if s_idx != prev_ind:
                    img_name = val_dataset.images[c_img_idx]
                    pred = predictions.loc[img_name][['p1', 'p2', 'p3', 'p4', 'p5']].to_list()
                    print(f"Sign predicted as:"
                          f" {' '.join([f'({i+1}): {val_dataset.idx_to_class[int(p)]}' for i, p in enumerate(pred)])}")
                    prev_ind = s_idx

            img = img.resize((200, 200))
            ab.xy = pos
            box_x = -150 if pos[0] > 0 else 150
            box_y = -100 if pos[1] > 0 else 100
            ab.xybox = (box_x, box_y)
            ab.offsetbox = OffsetImage(np.array(img), zoom=1.0)

        if not val:
            text = (" ".join([dataset.idx_to_class[lab.item()] for lab in y_data[ind["ind"]]]))
        elif correct:
            text = (" ".join([dataset.idx_to_class[lab.item()] for lab in y_data_val[correct_idx[ind["ind"]]]]))
        else:
            text = (" ".join([dataset.idx_to_class[lab.item()] for lab in y_data_val[wrong_idx[ind["ind"]]]]))

        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('r')
        annot.get_bbox_patch().set_alpha(0.4)

    # Handle hovering event
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == axes:

            cont, ind = sc.contains(event)
            if x_data_val is not None:
                cont_val, ind_val = sc_val_c.contains(event)
                cont_val_w, ind_val_w = sc_val_w.contains(event)
            else:
                cont_val, cont_val_w = False, False
                ind_val, ind_val_w = 0, 0

            if cont:
                update_annot(sc, ind, val=False)
                annot.set_visible(True)
                ab.set_visible(True)
                fig.canvas.draw_idle()
            elif cont_val:
                update_annot(sc_val_c, ind_val, val=True, correct=True)
                annot.set_visible(True)
                ab.set_visible(True)
                fig.canvas.draw_idle()
            elif cont_val_w:
                update_annot(sc_val_w, ind_val_w, val=True, correct=False)
                annot.set_visible(True)
                ab.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    ab.set_visible(False)
                    fig.canvas.draw_idle()

    # Clean-up axes, don't show ticks.
    for sp in ['right', 'top', 'bottom', 'left']:
        axes.spines[sp].set_visible(False)
    axes.tick_params('both', length=0, width=1)
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.grid(zorder=0)

    axes.set_xlim(-99, 99)
    axes.set_ylim(-99, 99)

    # Connect event handler and adjust margins
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.subplots_adjust(left=0.3, right=0.7, top=0.8, bottom=0.2)

    # Manually add legend because scatter plot doesn't make this easy
    patches = [matplotlib.patches.Patch(color=colors[i], label=name) for i, name in enumerate(cat_names)]
    plt.legend(handles=patches)




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('sign', nargs='*',
                        help='Sign names separate by spaces. If sign names include brackets, write sign names within '
                             'quotation marks.')
    parser.add_argument('--calc_features', action='store_true',
                        help='If set, calculate new features. Requires model weights, should not be used without '
                             'knowing what you are doing')
    parser.add_argument('-k', type=int,
                        help="Specify number of clusters in only one sign is shown. If not used, k is determined "
                             "automatically using the GAP-statistic.")
    parser.add_argument('--group', choices=['provenience'],
                        help="Optional group argument to group samples. Currently only tested with provenience")
    parser.add_argument('--exclude_val', action='store_true',
                        help="If set, only showing training set. Not recommended, as training set may be overfitted")
    args = parser.parse_args()


    # Data directories
    data_root = '../data_vat_o'
    split_file = 'train_val_split_cunes_hs_vat_o.txt'

    # data_root = '../data_test'
    # split_file = 'train_val_split_08_02_0411_1142.txt'

    img_type = 'SketchB'

    _, val_transform = get_transforms(img_type=img_type, resize=224)
    # _, val_transform = get_transforms(img_type=img_type, resize=128)


    if args.calc_features:
        train_set = CuneiformClassification(data_root, min_nb_img=20, split='train', transform=val_transform, lazy_load=True,
                                            img_type=img_type, split_file=split_file)
        val_set = CuneiformClassification(data_root, min_nb_img=20, split='val', transform=val_transform, lazy_load=True,
                                          img_type=img_type, split_file=split_file)

        exp_id = 37
        model_path = f'./results/{exp_id}/0325_120432/models/weights.pth'
        model = torch.load(model_path)
        for name, ds in zip(['train', 'val'], [train_set, val_set]):
            features, targets = calc_features(model, ds, 'cuda')
            torch.save(features, f'{exp_id}_{name}_features.pth')
            torch.save(targets, f'{exp_id}_{name}_targets.pth')

    else:
        exp_id = 37

        train_set = CuneiformClassification(data_root, min_nb_img=20, split='train', transform=None, lazy_load=True,
                                            img_type=img_type, split_file=split_file)
        val_set = CuneiformClassification(data_root, min_nb_img=20, split='val', transform=None, lazy_load=True,
                                          img_type=img_type, split_file=split_file)

        print(len(train_set), len(val_set))

        # Load train features
        features = torch.load(f'output/features/{exp_id}_train_features.pth', map_location='cpu')
        targets = torch.load(f'output/features/{exp_id}_train_targets.pth', map_location='cpu')

        # Load val features
        val_features = torch.load(f'output/features/{exp_id}_val_features.pth', map_location='cpu')
        val_targets = torch.load(f'output/features/{exp_id}_val_targets.pth', map_location='cpu')

        print(val_features.shape)

        # knn_classification(features, targets, val_features, val_targets, ds=val_set)

        # mean_distance(features, targets, train_set)
        # mean_distance(val_features, val_targets, val_set)

        # TSNE plot
        if args.exclude_val:
            tsne_plot(features, targets, train_set, args.sign, k=args.k)
        else:
            tsne_plot(features, targets, train_set, args.sign, val_features, val_targets, val_set, args.k,
                      group_key=args.group)

        plt.tight_layout()
        # plt.savefig(f'./graphs/tsne/tsne_{args.sign[0]}.svg')
        # plt.savefig(f'./graphs/tsne/tsne_ansze_giri.svg')
        plt.show()



if __name__ == '__main__':
    main()