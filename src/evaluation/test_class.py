import os
from collections import defaultdict

import pandas as pd
import torch.utils.data
from torch.utils.data import DataLoader


class Evaluator:
    """
    This class creates the results.csv file and takes care of evaluating, as well as keeping track of the c
    current number of iterations.

    It can store checkpoints to, but that's not fully tested.
    """

    def __init__(self, log_dir, result_name):
        self.log_dir = log_dir
        self.result_name = result_name
        self.metrics = defaultdict(list)

    def evaluate(self, model, val_ds, train_ds=None, print_metrics=False, device='cuda'):
        model = model.to(device)
        val_loader = DataLoader(val_ds, batch_size=64, num_workers=4, shuffle=True)

        val_loss, val_acc = test_dataset(model, val_loader, device)
        self.update_simple_metric('val_loss', val_loss)
        self.update_simple_metric('val_acc', val_acc)

        if train_ds is not None:
            train_loader = DataLoader(train_ds, batch_size=64, num_workers=4)
            train_loss, train_acc = test_dataset(model, train_loader, device)
            self.update_simple_metric('train_loss', train_loss)
            self.update_simple_metric('train_acc', train_acc)

        if print_metrics:
            print(f"Val. accuracy {val_acc*100:.2f}% | Val. loss {val_loss:.5f}", flush=True)

    def update_simple_metric(self, metric_name, value):
        self.metrics[metric_name].append(value)

    def dump_results(self):
        metrics = self.get_log_metrics()
        df = pd.DataFrame.from_dict(metrics)
        result_file = os.path.join(self.log_dir, f'{self.result_name}.csv')

        if os.path.exists(result_file):
            df.to_csv(os.path.join(self.log_dir, f'{self.result_name}.csv'), mode='a', sep=';', header=False, index=False)
        else:
            df.to_csv(os.path.join(self.log_dir, f'{self.result_name}.csv'), mode='w', sep=';', header=True, index=False)

        for k in self.metrics.keys():
            self.metrics[k] = []  # Don't just reset, like this we keep the order.

    def get_log_metrics(self):
        max_length = max([len(v) for v in self.metrics.values()])
        log_metrics = {}

        for k in self.metrics:
            if len(self.metrics[k]) == max_length:
                log_metrics[k] = self.metrics[k]
            else:
                # Pad with -1 if not long enough, although it should probably be empty
                log_metrics[k] = self.metrics[k] + [-1] * (max_length - len(self.metrics[k]))

        return log_metrics

def test_dataset(model, loader, device='cpu'):
    """
    Return the loss and accuracy on a single dataset (which is provided through a loader)
    Predicted logit is the argmax of the output.
    """
    loss, correct, length = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += torch.nn.functional.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            length += len(target)

    model.train()
    return loss.item() / length, correct / length

def test_topk_dataset(model, dataset, topk=(1, ), device='cpu'):
    model.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)
    correct_count, length = torch.zeros(len(topk)).to(device), 0
    max_k = max(topk)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            _, pred = output.topk(k=max_k, dim=1)
            pred = pred.T
            correct = pred.eq(target.view(1, -1)).expand_as(pred)

            for i, k in enumerate(topk):
                k_correct = correct[:k].sum(dim=0)
                correct_count[i] += sum(k_correct)
            length += len(correct[0])

    model.train()
    return correct_count / length

def test_dataset_per_class(model, loader: torch.utils.data.DataLoader, device='cpu'):
    losses, length, correct = defaultdict(float), defaultdict(float), defaultdict(float)
    model.eval()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target, reduction='none')
        pred = output.argmax(dim=1)

        for lo, pr, ta in zip(loss, pred, target):
            ta = ta.item()
            losses[ta] += lo.item()
            length[ta] += 1
            if pr.item() == ta:
                correct[ta] += 1

    model.train()
    return {label: losses[label] / length[label] for label in losses.keys()}, \
           {label: correct[label] / length[label] for label in losses.keys()}

def dump_pred(model, dataset, out_file='./out.csv', device='cpu', k=25):
    """
    Creates a file with the top-k predictions of model on dataset and logs them to out_file
    """
    model.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
    img_seen = 0
    predictions = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            _, pred = output.topk(k=k, dim=1)
            confidence = torch.nn.functional.softmax(output, dim=1)

            for t, p, c in zip(target, pred, confidence):
                c = c[p]
                pred_str = ';'.join([str(s.item()) for s in p])
                conf_str = ';'.join([f"{s.item():.5f}" for s in c])
                pred_str = f"{dataset.images[img_seen]};{t.item()};{pred_str};{conf_str}"
                predictions.append(pred_str)
                img_seen += 1

    with open(out_file, 'w') as f:
        header1 = ';'.join([f'p{i}' for i in range(1, k+1)])
        header2 = ';'.join([f'c{i}' for i in range(1, k+1)])
        f.write(f'img_file;target;{header1};{header2}\n')
        for line in predictions:
            f.write(line + '\n')

    model.train()
