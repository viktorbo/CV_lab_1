import logging as log
import time

import numpy as np


def rgb2gray(image: np.ndarray, mode: str):
    start_time = time.time()
    converter_name = 'rgb2gray'
    logger = log.getLogger(__name__ + '.gray_img')
    h, w = image.shape[0], image.shape[1]
    gray_img = np.zeros((h, w))

    mode_list = ['add', 'mid']
    assert mode in mode_list, logger.error('Wrong "mode" for {}: "{}" not in {}!'.format('"gray_img"', mode, mode_list))

    if mode == 'add':
        for i in range(h):
            for j in range(w):
                R = 0.2989
                G = 0.5870
                B = 0.1140
                gray_img[i][j] = R * image[i][j][0] + G * image[i][j][1] + B * image[i][j][2]

    elif mode == 'mid':
        for i in range(h):
            for j in range(w):
                gray_img[i][j] = sum(image[i][j]) / 3

    logger.info('Image was converted to grayscale and has shape {}'.format(gray_img.shape))
    logger.info('Convertion info:')
    logger.info('   Converter name: {}'.format(converter_name))
    logger.info('   Mode: {}'.format(mode))
    logger.info('   Time: {}'.format(time.time() - start_time))

    return gray_img


