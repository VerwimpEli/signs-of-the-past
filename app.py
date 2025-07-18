import urllib.request
from io import BytesIO
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from matplotlib import pyplot as plt

from PIL import Image

from src import *
from classification_tool import class_tool_transforms, load_idx_to_class

from gradio_image_annotation import image_annotator

class App:

    def __init__(self):
        self.model_path = './models/coco_vat_model.pth'
        self.idx_to_class_path = '../class_tool_files/misc/idx_to_class.txt'

        self.model = torch.load(self.model_path, map_location='cpu')
        self.model.eval()

        self.idx_to_class = load_idx_to_class(self.idx_to_class_path)
        self.val_transforms = class_tool_transforms('SketchB', 224)
        self.last_pred = None
        self.k = 5


    @torch.no_grad()
    def __call__(self, data, **kwargs):
        plt.close()
        img = data['image']
        pad_width = 0

        x1 = data['boxes'][0]['xmin']
        x2 = data['boxes'][0]['xmax']
        y1 = data['boxes'][0]['ymin']
        y2 = data['boxes'][0]['ymax']

        cy = (y1 + y2) // 2 + pad_width
        cx = (x1 + x2) // 2 + pad_width

        height = (x2 - x1) // 2
        width = (y2 - y1) // 2

        # if height == 0 or width == 0:
        #     return

        # Make selection square, because our models are trained using squares.
        if height > width:
            width = height
        else:
            height = width

        img_cutout = img[cy - height:cy + height, cx - width:cx + width]
        torch_img = torch.tensor(img_cutout).unsqueeze(0)
        torch_img = torch_img.permute(0, 3, 1, 2)

        torch_img = self.val_transforms(torch_img)
        output = self.model(torch_img)
        confidence = torch.nn.functional.softmax(output, dim=1)

        top_output = torch.topk(confidence, k=5)
        conf_values = top_output.values[0].numpy()
        pred = top_output.indices[0].numpy()

        self.last_pred = pred[0]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        visualize_single_pred(root='../class_tool_files/misc', axes=[axes[0], axes[1]], pred=pred[::-1],
                              conf=conf_values[::-1], img=img_cutout, idx_to_class=self.idx_to_class, color='#7a2d9a')
        print(f"Predicted: {pred[0]}")

        fig.tight_layout()
        return fig

    def reload_model(self):
        self.model = torch.load(self.model_path, map_location='cpu')
        self.model.eval()

def load_image(tablet, face, visualization, **kwargs):
    gradio_vis_to_iiif_vis = {
        'Sketch': 'sketchB',
        'ColorA': 'colorA',
    }

    iiif_base_link = "https://iiif.ghentcdh.ugent.be/iiif/images/cune-iiif-orm:wp5:ocr:{}:{}_{}_{}/full/full/0/default.jpg"
    full_url = iiif_base_link.format(tablet, tablet, gradio_vis_to_iiif_vis[visualization], face)

    with urllib.request.urlopen(full_url) as f:
        img = f.read()
    img = Image.open(BytesIO(img))

    return dict(image=img)


def load_sign_info(*args, **kwargs):
    global app
    sign_map = utils.load_cls_name_to_unicode_map(root='../data_vat_o')
    unicode_sign = sign_map[app.idx_to_class[app.last_pred]]

    return fr'''
            <iframe
                src="https://en.wiktionary.org/wiki/{unicode_sign}"
                name="targetframe"
                allowTransparency="true"
                scrolling="yes"
                frameborder="0"
                width="100%"
                height="500px"
            >
            </iframe>
            '''

def load_allowed_tablets(file: str):
    tablets = []
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if 'HS' in line:
                line = line.replace('HS_', 'HS')

            if not 'O_' in line:
                tablets.append(line)
    return tablets

def change_model(value, **kwargs):
    global app

    if value == 'Sketch':
        app.val_transforms = class_tool_transforms('SketchB', 224)
        app.model_path = './models/coco_vat_model.pth'
    elif value == 'ColorA':
        app.val_transforms = class_tool_transforms('ColorA', 224)
        app.model_path = './models/coco_color_model.pth'

    app.reload_model()


theme = gr.themes.Soft(font=["ui-sans-serif", "Arial", "sans-serif"])
gr.set_static_paths(paths=[f'{Path.cwd().absolute()}/static'])

with gr.Blocks(theme=theme) as demo:
    app = App()

    html_header = gr.HTML(r'''
        <h1  style="text-align: center; font-size: 72px"> Cune-AI-form </h1>
    ''')

    markdown_explanation = gr.Markdown(r'''
    ### Introduction
    The Cune-AI-form assistant will predict which cuneiform sign is selected by the user on a tablet. These can be the
    tablets that are recorded as part of this project, or one that you upload yourself. Our machine learning model will
    propose the 5 most likely sign classifications, unless it is certain, then there will be fewer.
    The models are trained on Old Babylonian tablets, on either our generated sketches ('Sketch'), or light coming from the top 
    ('ColorA'). Predictions will therefore be best if you have similar tablets recorded in similar conditions.
    
    For more details, see our paper: [COMING SOON]. \
    *Note: the only difference is a different train/val split in this tool. Here tablets are either completely in the
    train or in validation set, whereas in the paper they can be mixed.*
    
    ### Guide
    1. Load a tablet. There are two options: upload your own or load one of ours.
        1. To load one of ours: select the tablet, the appropriate face and the visualizations and click 'load'.
        2. To use your own: click the image region below the 'predict' button. If another image is loaded, first delete
        it by clicking the cross in the top right.
    2. Select the correct model. The 'Sketch' model will not work on color images and vice-versa.
    2. Draw a rectangle. 
        1. Select the 'create box' tool at the bottom of the image and optionally modify it with the 'drag box' tool.
        2. Shortcuts are a convenient way to draw new boxes (c), modify the current box (d) or delete the current box (del)
    3. Make the prediction by clicking the predict button. The results will appear on the right, together with a cut-out
    of the selected sign.
    4. (Optional). Look up the most likely sign on Wiktionary by clicking the 'look up' button.
    
    '''
    )

    logos = gr.HTML(r'''
        <img style="margin: 10px auto 20px; display: block; width: 40%;" src='file=./static/all_logo.jpg'>
    ''')

    with gr.Row():
        dropdown = gr.Dropdown(label='Choose tablet', info='You can also upload a personal image below. The ones '
                                                           'that you can select here are in our validation set.',
                               choices=load_allowed_tablets('../class_tool_files/misc/val_tablets.txt'),
                               scale=2)
        face_btn = gr.Dropdown(choices=['front', 'back', 'left', 'right', 'top', 'bottom'], scale=1,
                               label='Tablet face', info="Not all faces are always available, image won't load"
                                                         "if face is not available")
        vis_btn = gr.Radio(choices=['Sketch', 'ColorA'], scale=1, label='Tablet visualization',
                             info='Make sure to select the correct model as well', value='Sketch')

        load_btn = gr.Button("Load tablet")

    model_btn = gr.Radio(label='Model', info='Pick data that the prediction model is trained on. Should be similar to '
                                             'the image that you load here',
                                             choices=['Sketch', 'ColorA'], value='Sketch')

    with gr.Row():
        with gr.Column():
            sub_btn = gr.Button("Predict", scale=1)
            img_annot = image_annotator(show_label=False, single_box=True, disable_edit_boxes=True, min_width=50)
            spacer = gr.HTML()

        with gr.Column():
            plot = gr.Plot(label='Prediction')
            lookup_button = gr.Button("Look up", scale=0)
            info_panel = gr.HTML(r'''
            Hit the look up button to look up the most likely prediction on Wiktionary.
            ''')


    sub_btn.click(app, inputs=img_annot, outputs=plot)
    load_btn.click(load_image, inputs=[dropdown, face_btn, vis_btn], outputs=img_annot)
    lookup_button.click(load_sign_info, inputs=None, outputs=info_panel)
    model_btn.input(change_model, model_btn)

    gr.Markdown("*Author tool: Eli Verwimp*")


demo.launch(share=False, max_file_size='10mb')
