import runway
import numpy as np
import argparse
import torch
from run import restore

@runway.command('translate', inputs={'source_imgs': runway.image(description='input image to be translated'),}, outputs={'image': runway.image(description='input image to be translated')})
def translate(model, inputs):
    image = restore(inputs['source_imgs'], True)
    return image


if __name__ == '__main__':
    runway.run(port=8889)