import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
import torchvision
from torchvision.transforms import v2
import torch
import torch.utils.data

from src import *

def get_transforms(img_type='SketchB', augmentations: str = None, resize: int = 128):

    mean, std = get_image_statistics('grayscale' if 'grayscale' == augmentations else img_type, log=True)

    base_transforms = [
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=resize, antialias=True),
        v2.Normalize(mean, std),
    ]

    if augmentations == 'rot':
        print("Using rotation augmentations")
        train = base_transforms + [v2.RandomApply([v2.RandomRotation((0, 360))], 0.5)]
        test = base_transforms
    elif augmentations == 'rot_pers':
        print("Using rotation/perspective augmentations")
        train = base_transforms + [v2.RandomApply([v2.RandomRotation((0, 360))], 0.5),
                                   v2.RandomPerspective(distortion_scale=0.5, p=0.5)]
        test = base_transforms
    elif augmentations == 'debug3':
        print("Using debug3 augmentations")
        train = base_transforms + [v2.RandomApply([v2.RandomRotation((0, 360))], 0.5),
                                   v2.RandomPerspective(distortion_scale=0.5, p=0.5)]
        test = base_transforms
    elif augmentations == 'grayscale':
        print("Using grayscale augmentations")
        base_transforms.insert(2, v2.Lambda(lambda x: (1/3 * x[0] + 1/3 * x[1] + 1/3 * x[2])
                                            .unsqueeze(0).repeat(3, 1, 1)))
        train = base_transforms + [v2.RandomApply([v2.RandomRotation((0, 360))], 0.5),
                                   v2.RandomPerspective(distortion_scale=0.5, p=0.5)]
        test = base_transforms
    else:
        print("Using base transforms")
        train, test = base_transforms, base_transforms

    train_transforms = v2.Compose(train)
    test_transforms = v2.Compose(test)

    return train_transforms, test_transforms

def get_data(root,  split_file: str, img_type: str, test_img_type: str = None, min_nb_img: int = 20, train_transform = None,
             test_transform = None, lazy_load = False, **kwargs):
    if test_img_type is None:
        test_img_type = img_type

    train_set = CuneiformClassification(root, min_nb_img=min_nb_img, split='train', transform=train_transform, lazy_load=lazy_load,
                                        img_type=img_type, split_file=split_file, **kwargs)
    val_set = CuneiformClassification(root, min_nb_img=min_nb_img, split='val', transform=test_transform, lazy_load=lazy_load,
                                      img_type=test_img_type, split_file=split_file, **kwargs)
    return train_set, val_set

def get_model(model: str, num_classes: int, device='cuda', init_model: str = None):

    if init_model is not None:
        # Models are stored as pickle objects, so no state dicts are necessary here.
        return torch.load(init_model)

    if model == 'resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif model == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    elif model == 'resnext50':
        model = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
    elif model == 'vit':
        raise ValueError(f"Not implemented yet")
    else:
        raise ValueError(f"Unknown model: {model}")

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    return model

def find_init_model(init_model_exp_id: str, log_dir: str):
    """
    Picks a model from the experiments in the init_model_exp_id result folders. It writes to picked model to
    'init_models.txt' in the top level experiment folder. Next time this is run, this file will be checked and
    a different model that has been used less is picked.

    The picked file is added to the init_models.txt file. If an experiment crashes this won't be removed, so the
    true count may be lower.
    :param init_model_exp_id: str that refers to the init model. If NaN, None is returned
    :param log_dir: log_dir of this experiment
    :return: the path of the weights that need to be loaded.
    """
    # Log dir should contain file that has all models that are already tried.

    if pd.isna(init_model_exp_id):
        return None
    else:
        # Cast to int and then string here, it'll be a float before because pandas int columns can't have nan's
        init_model_exp_id = str(int(init_model_exp_id))

    exp_path = os.path.join('results', init_model_exp_id)
    exp_dirs = [ed for ed in os.listdir(exp_path) if not 'running.txt' in os.listdir(os.path.join(exp_path, ed))]


    file_path = os.path.join(log_dir, '../init_models.txt')
    if not os.path.exists(file_path):
        with open(file_path, 'x') as f:
            pass

    used_exp_dir_counts = {k: 0 for k in exp_dirs}
    # Make sure to read all lines first if not writing happens in the middle of the file.
    with open(file_path, 'r+') as f:
        for line in f.readlines():
            _, line = line.strip().split(' ')
            used_exp_dir_counts[line] += 1

        picked_dir = min([(k, v) for k, v in used_exp_dir_counts.items()], key=lambda x: x[1])
        f.write(f"{log_dir} {picked_dir[0]}\n")

    return os.path.join('./results', init_model_exp_id, picked_dir[0], 'models', 'weights.pth')

def get_optim_and_scheduler(model: torch.nn.Module, optim: str, sched: str, num_epochs: int):
    if optim == 'adam':
        optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    elif optim == 'adam_ft':
        print("Using adam_ft")
        optim = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0)
    else:
        raise ValueError(f"Unknown optimizer: {optim}")

    if sched == 'cos':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs, eta_min=1e-5)
    elif sched == 'cos_ft':
        print("Using cos_ft")
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs, eta_min=1e-7)
    else:
        raise ValueError(f"Unknown scheduler: {sched}")

    return optim, sched

def get_img_types(exp_settings):
    img_types = exp_settings['img_type'].split(' ')
    img_types = img_types if len(img_types) > 1 else img_types[0]

    test_img_types = exp_settings['test_img_type']
    if pd.isna(test_img_types):
        test_img_types = img_types
    else:
        test_img_types = test_img_types.split(' ')
        test_img_types = test_img_types if len(test_img_types) > 1 else test_img_types[0]

    return img_types, test_img_types

def get_proveniences(exp_settings):
    provenience = exp_settings['provenience'].split(' ')
    return provenience

def train_model(model, optim, scheduler, train_ds, val_ds, evaluator, exp_settings, device):
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(exp_settings['batch_size']),
                                               num_workers=4, shuffle=True)
    epochs = exp_settings['num_epochs']

    for e in range(epochs):
        print(f"Epoch {e}")
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optim.step()
        scheduler.step()

        evaluator.evaluate(model, val_ds, device=device, print_metrics=True)

    print(test_dataset(model, train_loader, device=device))

def run_exp(exp_id: int):
    """
    Start experiment based on a given exp_id and the experiments described in experiments.csv
    :return:
    """

    device = 'cpu'
    root = '../data_vat_o'
    # split_file = 'train_val_split_cunes_hs_vat_o.txt'
    split_file = 'coco_tablet_split.txt'

    exp_file = pd.read_csv('./exp_configurations/class_experiments.csv',
                           sep=';', index_col='exp_id', dtype={'batch_size': int})
    exp_settings = exp_file.loc[exp_id]
    log_dir = init_logging(exp_settings)

    print(exp_settings)
    print(f"Using split file: '{split_file}'")

    # Prepare data
    img_types, test_img_types = get_img_types(exp_settings)
    proveniences = get_proveniences(exp_settings)
    train_transform, test_transform = get_transforms(img_types, exp_settings['augmentations'], int(exp_settings['img_size']))
    train_ds, val_ds = get_data(root, split_file, img_types, test_img_types, exp_settings['min_nb_img'],
                                train_transform, test_transform, lazy_load=True, tablet_type=proveniences)
    print(f"Length train DS: {len(train_ds)} | val DS {len(val_ds)}")

    # Get model initialization
    num_classes = len(set(train_ds.labels))
    init_model_path = find_init_model(exp_settings['init_model'], log_dir)
    print(f"Initializing model: {init_model_path}")
    model = get_model(exp_settings['model'], num_classes, device, init_model_path)

    # Init evaluator.
    evaluator = Evaluator(log_dir, result_name='results')

    # Get optim and scheduler.
    num_epochs = exp_settings['num_epochs']
    optim, scheduler = get_optim_and_scheduler(model, exp_settings['optim'], exp_settings['sched'], num_epochs)

    evaluator.evaluate(model, val_ds, device=device, print_metrics=True)

    # Actual training.
    train_model(model, optim, scheduler, train_ds, val_ds, evaluator, exp_settings, device)

    # Store predictions, results and model weights
    evaluator.dump_results()
    torch.save(model, os.path.join(log_dir, 'models', 'weights.pth'))

    # Dump predictions with lazily loaded dataset to get image file names
    _, val_ds = get_data(root, split_file, img_types, test_img_types, exp_settings['min_nb_img'],
                         train_transform, test_transform, lazy_load=True)
    dump_pred(model, val_ds, out_file=os.path.join(log_dir, 'predictions.csv'), device=device, k=25)

    # If we get here, the program finished successfully, and we can remove this file.
    os.remove(os.path.join(log_dir, 'running.txt'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_id', type=int, help='the exp_id from the experiments.csv file')
    parser.add_argument('-r', '--reps', type=int, default=1, help='number of repetitions to run')
    args = parser.parse_args()

    for _ in range(args.reps):
        run_exp(args.exp_id)


if __name__ == '__main__':
    main()
