import runway
import numpy as np
import argparse
import torch
from run import restore
from PIL import Image

@runway.command('translate', inputs={'source_imgs': runway.image(description='input image to be translated'),'scratch_remove': runway.boolean(default=True),}, outputs={'image': runway.image(description='input image to be translated')})
def translate(model, inputs):
    image = restore(inputs['source_imgs'], inputs['scratch_remove'])
    im = Image.open(open(image, 'rb'))
    return im

if __name__ == '__main__':
    runway.run(port=8889)