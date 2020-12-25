import runway
import numpy as np
import argparse
import torch


@runway.command('translate', inputs={'source_imgs': runway.image(description='input image to be translated'),'amount': runway.number(min=0, max=100, default=0,description='Age'), 'gender': runway.boolean(default=True,description='On uses male model, off uses female model'),}, outputs={'image': runway.image(description='input image to be translated')})
def translate(model, inputs):
    return blah


if __name__ == '__main__':
    runway.run(port=8889)