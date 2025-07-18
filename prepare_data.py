from src import *

# This method will create the sign crops on which our models are trained. For more details see the documentation
# see the documentation of create_sign_crops, in ./src/data/prepare_class_data.py

root = '../data'
img_type = 'SketchB'

create_sign_crops(root=root, img_types=[img_type])

