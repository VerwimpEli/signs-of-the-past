import torch
from torch.utils.data import DataLoader
from torchvision.ops import boxes

from .coco_eval import CocoEvaluator
from .test_class import Evaluator

# TODO: this is not necessarily best here, but it is necessary here and not in other packages. Only thing is that
# it is also necessary as a function, so if I manually expose some methods, this should definitely be in it.
def default_collate(batch):
    return tuple(zip(*batch))

class DetectionEvaluator(Evaluator):

    base_metrics = ['AP @0.5:0.95', 'AP @0.5', 'AP @0.75', 'AP small', 'AP medium', 'AP large',
                    'AR 1d', 'AR 10d', 'AR 100d', 'AR small', 'AR medium', 'AR larger',
                    'AR @0.7 200', 'AP @0.7 200d']

    def __init__(self, log_dir, result_name):
        super().__init__(log_dir, result_name)

    def evaluate(self, model, val_ds, train_ds=None, print_metrics=False, device='cuda'):
        model.to(device)
        model.eval()
        loader = DataLoader(val_ds, batch_size=2, collate_fn=default_collate, num_workers=4)
        results = evaluate(model, loader, device=device)

        for name, r in zip(self.base_metrics, results['bbox']):
            self.update_simple_metric(name, r)


# TODO: This expects a DetectionDS: perhaps this should go to data: this is not necessarily evaluation
# related.
def overwrite_coco_category_ids(ds):
    for ann in ds.coco.anns:
        if ds.single_label:
            ds.coco.anns[ann]['category_id'] = 1
        else:
            ds.coco.anns[ann]['category_id'] = ds.class_map[ds.coco.anns[ann]['category_id']]


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()

    # FIXME remove this and make paste_masks_in_image run on the GPU (this is not from me, Eli)
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    # TODO: this should may be not be here?
    overwrite_coco_category_ids(data_loader.dataset)

    coco = data_loader.dataset.coco
    coco_evaluator = CocoEvaluator(coco, ['bbox'])

    # TODO: make this more like the classification logging
    # log_file = './log_detection_test_1000p_1250ms.csv'
    # with open(log_file, 'w') as f:
    #     f.write('image_id;pred;conf;x0;y0;x1;y1\n')

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)

        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        for prediction in outputs:
            keep_idx = boxes.nms(prediction['boxes'], prediction['scores'], 0.8)
            for key in prediction:
                prediction[key] = prediction[key][keep_idx]

        # with open(log_file, 'a') as f:
        #     for target, prediction in zip(targets, outputs):
        #         for i in range(len(prediction['labels'])):
        #             log_values = [prediction['labels'][i],  prediction['scores'][i], *prediction['boxes'][i]]
        #             log_values = [target['image_id']] + [x.item() for x in log_values]
        #             f.write(';'.join([str(l) for l in log_values]) + '\n')

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    results = coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    return results

