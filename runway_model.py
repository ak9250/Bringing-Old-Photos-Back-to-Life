import runway
import numpy as np
import argparse
import torch
from torchvision import transforms
import os.path
import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import util.preprocess_itw_im as preprocess
import util.deeplab as deepl
import numpy as np
import os
import cv2
import time
import unidecode       
from PIL import Image

opt = TestOptions().parse(save=False)
opt.display_id = 0 # do not launch visdom
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.in_the_wild = True # This triggers preprocessing of in the wild images in the dataloader
opt.traverse = True # This tells the model to traverse the latent space between anchor classes
opt.interp_step = 0.05 # this controls the number of images to interpolate between anchor classes



@runway.command('translate', inputs={'source_imgs': runway.image(description='input image to be translated'),'amount': runway.number(min=0, max=100, default=0,description='Age'), 'gender': runway.boolean(default=True,description='On uses male model, off uses female model'),}, outputs={'image': runway.image(description='input image to be translated')})
def translate(model, inputs):
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    if inputs['gender']:
        opt.name = 'males_model'
    else:
        opt.name = 'females_model'         
    model = create_model(opt) 
         
    os.makedirs('images', exist_ok=True)
    inputs['source_imgs'].save('images/temp.jpg')
    paths = os.path.join('images','temp.jpg')
    
    data = dataset.dataset.get_item_from_path(paths)
    visuals = model.inference(data)

    visual = visuals[0]
    orig_img = visual['orig_img']
    out_classes = len(visual) - 1
    slide = inputs['amount']
    next_im = visual['tex_trans_to_class_' + str(slide)]
    return next_im


if __name__ == '__main__':
    runway.run(port=8889)