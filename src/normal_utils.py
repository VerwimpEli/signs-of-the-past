import json
import os
import time

import PIL.Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression

from .data.utils import extract_points_from_svg, polygon_to_bbox, get_annotations


def visualize_normals(image_path: str, annot_path: str):
    """
    Visualize the average normal of annotations of a tablet, provided the path to the json annotations and the image.
    Easy to use this to see how the normals are not pointed in the correct directions.
    """
    fig, axes = plt.subplots()

    image = PIL.Image.open(image_path)
    image = np.array(image)
    axes.imshow(image)

    arrow_size = 100

    with open(annot_path, 'r') as f:
        data = json.load(f)

    for i, key in enumerate(data):
        svg_str = data[key]['target']['selector']['value']

        coordinates = extract_points_from_svg(svg_str, return_int=True)
        bbox = polygon_to_bbox(coordinates)

        bw, bh = bbox[1, 0] - bbox[0, 0], bbox[1, 1] - bbox[0, 1]
        axes.add_patch(Rectangle((bbox[0, 0], bbox[0, 1]), bw, bh, fill=False, lw=2, edgecolor='white'))

        cx = int(1 / 2 * (bbox[0, 0] + bbox[1, 0]))
        cy = int(1 / 2 * (bbox[0, 1] + bbox[1, 1]))

        avg_normal = np.mean(image[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]], axis=(0, 1))
        avg_normal = avg_normal / 255 * 2 - 1

        axes.arrow(cx, cy, avg_normal[1] * arrow_size, avg_normal[0] * arrow_size, head_width=30)

    plt.show()


def detect_normal_orientation(image: np.ndarray, plot: bool = False) -> int:
    """
    Detects the orientation of the image's normal map, assuming the z-direction is pointing upwards. Only works
    with imags of a single faces (i.e., not in fat-cross format).
    :param image:  numpy array as HxWxC, where the channel axis has dimension 3 (RGB)
    :param plot: if True, estimated axis and values will be shown, easy to see what went wrong.
    :return: integer rotation of the normal map.
    """

    # Mapping from axis order and slope (True = positive) to the rotated angle.
    rotation_dir = {
        (0, True, 1, False): 0,
        (1, True, 0, True): 90,
        (0, False, 1, True): 180,
        (1, False, 0, False): 270,
    }

    # Estimate the direction of red channel
    x_mean = np.mean(image[..., 0], axis=0)
    y_mean = np.mean(image[..., 0], axis=1)

    # Check along which axis the minimal and maximal values lay. The correct direction will on average contain
    # the largest minimum and maximal values on an axis
    color_max = np.argmax([np.max(x_mean), np.max(y_mean)])
    color_min = np.argmin([np.min(x_mean), np.min(y_mean)])

    # The minimum and maximum average values do not occur on the same direction. If so, the image needs to be manually checked
    if color_max != color_min:
        raise ValueError('Red color axis not consistent. Exiting...')

    red_axis = color_max

    # Now that we have the axis, estimate the direction (i.e., what the positive side is) using linear regression on
    # the average values of that axis.
    reg = LinearRegression()
    reg.fit(np.arange(len([x_mean, y_mean][red_axis]))[:, None], [x_mean, y_mean][red_axis])
    red_co = reg.coef_[0]

    # Do the same operations for the green channel.
    x_mean = np.mean(image[..., 1], axis=0)
    y_mean = np.mean(image[..., 1], axis=1)

    color_max = np.argmax([np.max(x_mean), np.max(y_mean)])
    color_min = np.argmin([np.min(x_mean), np.min(y_mean)])

    if color_max != color_min:
        raise ValueError('Green color axis not consistent. Exiting...')

    green_axis = color_max

    reg = LinearRegression()
    reg.fit(np.arange(len([x_mean, y_mean][green_axis]))[:, None], [x_mean, y_mean][green_axis])
    green_co = reg.coef_[0]

    # They green and red axis should be different, so if they are not, the image needs to be manually checked.
    if green_axis == red_axis:
        raise ValueError("Green dir equals red dir")

    # Plot mean colors and estimated directions
    if plot:
        fig, axes = plt.subplots(2, 3)
        for i in range(3):
            axes[0, i].imshow(image[..., i])

            x_mean = np.mean(image[..., i], axis=0)
            y_mean = np.mean(image[..., i], axis=1)

            x = np.arange(len(x_mean))[:, None]
            reg = LinearRegression().fit(x, x_mean)

            axes[1, i].plot(x_mean, label='x')
            axes[1, i].plot(reg.predict(x), label='x_pred')

            y = np.arange(len(y_mean))[:, None]
            reg = LinearRegression().fit(y, y_mean)

            axes[1, i].plot(y_mean, label='y')
            axes[1, i].plot(reg.predict(y), label='y_pred')

        plt.legend()
        plt.show()

    # Try to get the rotation from the rotation dir. If this is not found, the parameters are not estimated correctly
    # and manual verification is necessary
    try:
        rot_dir = rotation_dir[(red_axis, red_co >= 0, green_axis, green_co >= 0)]
    except KeyError:
        raise ValueError(f"Key not found: {(red_axis, red_co >= 0, green_axis, green_co >= 0)}")

    return rot_dir


def rotate_image(image: np.ndarray, rotated_angle: int) -> np.ndarray:
    """
    Rotate a normal map back to 0 degrees, with the red channel on the x-axis and the green axis on the y-axis. This
    only works correctly if the normal map is rotated, not flipped. If jpgs are used compression artifacts
    will become visible, so it is best to use pngs instead.

    :param image: numpy array as HxWxC, where the channel axis has dimension 3 (RGB) and values are integers between
    0 and 255.
    :param rotated_angle: the angle with which the normal map is currently rotated.
    :return: a copy of the rotated image.
    """

    image = image.copy()  # copy image else the original will be rotated as well.

    if rotated_angle == 0:
        pass
    elif rotated_angle == 90:
        image[..., [0, 1]] = image[..., [1, 0]]
        image[..., 1] = 255 - image[..., 1]
    elif rotated_angle == 180:
        image[..., 1] = 255 - image[..., 1]
        image[..., 0] = 255 - image[..., 0]
    elif rotated_angle == 270:
        image[..., [0, 1]] = image[..., [1, 0]]
        image[..., 0] = 255 - image[..., 0]
    else:
        raise ValueError("No 90 degree rotation")

    return image


def rotate_all_images(root: str):
    """
    Rotate all images to have a consistent normal orientation: red on the x-axis, with positive to the right and green
    on the y-axis with positive to the top. Images are saved on the same location, but replacing 'normal' with 'normalC'
    Images expected to be in subfolders per direction:
        - Root
            - e.g., CUNES51-01-01:
                - e.g., Normal_front.png
                - ...
            - ...
    :param root: root direction of the images
    """
    img_folders = sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))])

    for img_fold in img_folders:
        print(f"Rotating {img_fold}...")
        images = sorted(os.listdir(os.path.join(root, img_fold)))

        for img in images:
            img_path = os.path.join(root, img_fold, img)
            img = np.array(PIL.Image.open(img_path))

            try:
                rot_dir = detect_normal_orientation(img)
                rotated_image = rotate_image(img, rot_dir)
            except ValueError as e:
                print(img_fold, img_path, e)
                continue

            rotated_image = PIL.Image.fromarray(rotated_image)
            new_image_path = img_path.replace('normal', 'normalC')
            rotated_image.save(new_image_path)


def normal_to_sketch_opengl(image: np.ndarray, w: int):
    """
    Normal to sketch, original OpenGL implementation
    """

    start = time.perf_counter()

    normals = (image / 255) * 2 - 1
    image_result = np.zeros_like(normals)


    for x_co in range(w, image.shape[0] - w):
        for y_co in range(w, image.shape[1] - w):

            avg_normal = np.zeros(3)
            for i in range(1, w+1):
                for y in range(-i, i+1, i):
                    for x in range(-i, i+1, i):
                        neighbour = normals[x_co+x, y_co+y]
                        avg_normal += neighbour
            avg_normal /= np.linalg.norm(avg_normal)

            sigma2 = 0

            for i in range(1, w+1):
                for y in range(-i, i+1, i):
                    for x in range(-i, i+1, i):

                        neighbour = normals[x_co+x, y_co+y]
                        diff = avg_normal - neighbour

                        sign = np.dot(diff, np.array([x, y, 0.0]))
                        sigma2 = sigma2 + sign * np.dot(diff, diff)


            sigma2 /= (2*w+1) * (2*w+1)
            if sigma2 <= -0.01:
                wedge = 1.0
            elif sigma2 < 0.0:
                t = (sigma2 - 0.0) / (-0.01 - 0.0)  # t will be between 0 and 1
                wedge = t * t * (3.0 - 2.0 * t)
            else:
                wedge = 0

            wedge2 = 1.0 - wedge ** 0.1
            image_result[x_co, y_co] = wedge2

    stop = time.perf_counter()
    print(f"Old calculation took {stop-start:.3f} seconds")
    return image_result


def normal_to_sketch(image: np.ndarray):
    """
    Transform the normal map to sketch, sped up python implementation.
    """

    width, height = image.shape[0:2]
    width += 1000
    size = 200
    image = image[width // 2 - size:width // 2 + size, height // 2 - size:height // 2 + size]

    start = time.perf_counter()
    normals = (image / 255) * 2 - 1

    image_result = np.zeros_like(normals)

    w = 3

    mask = np.eye(2*w + 1)
    mask += np.fliplr(np.eye(2*w+1))
    mask[w] = 1
    mask[:, w] = 1
    mask[w, w] = w
    avg_mask = np.repeat(mask[..., None], 3, axis=2)

    y_s, x_s = np.meshgrid(np.arange(2*w+1), np.arange(2*w+1))
    sign_matrix = np.array([x_s - 3, y_s - 3, np.zeros_like(x_s)])
    sign_matrix = np.moveaxis(sign_matrix, 0, 2)

    diff_mask = mask.copy()
    diff_mask[w, w] = 0

    for x_co in range(w, image.shape[0] - w):
        for y_co in range(w, image.shape[1] - w):

            avg_normal = np.sum(avg_mask * normals[x_co-w:x_co+w+1, y_co-w:y_co+w+1], axis=(0, 1))
            avg_normal /= np.linalg.norm(avg_normal)


            diff_matrix = avg_normal - normals[x_co-w:x_co+w+1, y_co-w:y_co+w+1]
            # Calculate the dot product between the different normal and the sign_matrix. Positive values point outwards
            # from the center, negative values point inwards
            signs = np.einsum('ijk,ijk->ij', diff_matrix, sign_matrix)

            diff_norms = np.einsum('ijk,ijk->ij', diff_matrix, diff_matrix)
            sigma2 = np.average(diff_mask * signs * diff_norms)

            image_result[x_co, y_co] = sigma2


    def smooth_clip(x):
        clip_min = -0.005
        if x <= clip_min:
            return 1.0
        elif x > 0:
            return 0
        else:
            t = (x - 0.0) / (clip_min - 0.0)  # t will be between 0 and 1
            return t * t * (3.0 - 2.0 * t)

    vec_clip = np.vectorize(smooth_clip)
    image_result = vec_clip(image_result)

    image_result = 1.0 - image_result ** 0.1

    stop = time.perf_counter()
    print(f"Took {stop - start:.3f} seconds")

    # old_result = normal_to_sketch_opengl_copy(image, w)
    old_result = image_result

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(image)
    axes[0, 1].imshow(image_result)
    axes[1, 0].imshow(old_result)

    diff = image_result - old_result
    diff -= diff.min()
    diff /= diff.max()

    axes[1, 1].imshow(diff)

    for ax in axes.flatten():
        ax.axis('off')

    # img_res = PIL.Image.fromarray((image_result * 255).astype(np.uint8))
    # img_res.save('./my_sketch.png')

    plt.tight_layout()
    plt.show()


def get_normal_meta_data(root: str):
    """
    Calculate the average normal of all signs and add it to the metadata of a dataset.
    :return:
    """

    annotations = get_annotations(root)
    all_normals = []

    meta_data = pd.read_csv('../data_vat_o/sign_meta_data.csv', sep=';')
    meta_data['nx'] = np.NAN
    meta_data['ny'] = np.NAN
    meta_data['nz'] = np.NAN

    for i, annot in enumerate(annotations):
        print(annot)

        tablet_name, face = annot.rsplit('_', maxsplit=1)
        face = face.split('.')[0]

        if 'HS' in tablet_name:
            img_tablet_name = tablet_name.replace("HS_", "HS")
        else:
            img_tablet_name = tablet_name

        img_type = 'normalC'
        img_file = f'{root}/images/{img_tablet_name}/{img_tablet_name}_{img_type}_{face}.png'

        try:
            image = np.array(PIL.Image.open(img_file))
        except FileNotFoundError:
            continue

        with open(os.path.join(root, 'annotations', annot), 'r') as f:
            data = json.load(f)

        for _, key in enumerate(data):

            svg_str = data[key]['target']['selector']['value']

            try:
                coordinates = extract_points_from_svg(svg_str, return_int=True)
            except ValueError:
                continue
            bbox = polygon_to_bbox(coordinates)

            avg_normal = np.mean(image[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]], axis=(0, 1))
            avg_normal = avg_normal / 255 * 2 - 1
            all_normals.append(avg_normal)

            meta_data.loc[(meta_data['tablet'] == tablet_name) & (meta_data['sign_id'] == key), ['nx', 'ny', 'nz']] = avg_normal


    meta_data.to_csv('../data_vat_o/sign_meta_data_w_normal.csv', sep=';', index=False)


def main_normals():

    image_path = '../data_png/images/CUNES51-01-13/CUNES51-01-13_normal_front.png'
    annot_path = '../data_vat_o/annotations/CUNES51-01-13_front.json'

    image = np.array(PIL.Image.open(image_path))

    # Detect the orientation of the normal map.
    angle = detect_normal_orientation(image, plot=True)
    print(f'Estimated rotation angle: {angle} degrees')

    # Visualize the normal maps.
    # visualize_normals(image_path, annot_path)

    # Manually rotate an image for those where it is not possible to do so automatically and save.

    # rotated_image = rotate_image(image, 90)
    # rotated_image = PIL.Image.fromarray(rotated_image)
    # new_image_path = image_path.replace('normal', 'normalC')
    # rotated_image.save(new_image_path)

