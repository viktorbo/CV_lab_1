import logging as log
import os
from pathlib import Path

import numpy as np
from PIL.Image import open

def read_img_as_arr(path: str):
    logger = log.getLogger(__name__ + '.read_img_as_arr')
    image_path = str(Path(path))
    image = None
    try:
        image = np.array(open(image_path))
        logger.info('Image was loaded from {} as array with shape {}'.format(os.path.basename(image_path), image.shape))
    except FileNotFoundError:
        logger.error('Error! File {} not found!'.format(image_path))

    return image
