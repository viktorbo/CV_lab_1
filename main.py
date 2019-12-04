import argparse
import logging as log
import os
from copy import copy
from datetime import datetime

import cv2 as cv
import numpy as np

log.basicConfig(level=log.DEBUG,
                format=('{} [%(levelname)s] %(message)s').format(str(datetime.now().strftime('%d-%m-%Y %H:%M:%S'))))


def fix_contrast(img: np.ndarray):
    logger = log.getLogger('fix_contrast')
    min = np.amin(img)
    max = np.amax(img)
    k = 255 / (max - min)

    logger.info('Run fix_contrast converter')
    logger.info('MIN value: {}'.format(min))
    logger.info('MAX value: {}'.format(max))
    logger.info('255 / (MAX - MIN): {}'.format(k))
    try:
        h, w = img.shape
    except ValueError:
        logger.error('Wrong shape of image (image has {} channels instead of 1). Converter was not applied!'.format(img.shape[2]))
        return img

    tmp = copy(img)
    for i in range(h):
        for j in range(w):
            tmp[i][j] = (tmp[i][j] - min) * k

    return tmp

def show_edges(img: np.ndarray, edges: np.ndarray, size: int = 1):
    logger = log.getLogger('show_edges')
    try:
        h, w = img.shape
    except ValueError:
        logger.error('Wrong shape of image (image has {} channels instead of 1).'.format(img.shape[2]))
        return

    tmp = copy(img)
    r = size // 2
    for i in range(r, h-r):
        for j in range(r, w-r):
            if edges[i][j] != 0:
                for x in range(i-r, i+r+1):
                    for y in range(j-r, j+r+1):
                        tmp[x][y] = 0

    cv.imshow('image with edges', tmp)


def main():
    logger = log.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Path to input image')
    args = parser.parse_args()

    if args.input is None:
        default_file_name = 'tulips.jpg'
        default_path = os.path.abspath(default_file_name)
        logger.warning('The default input file will be used.')
        input_path = default_path
    else:
        input_path = os.path.abspath(args.input)

    assert os.path.exists(input_path), logger.error('Input file {} does not exist!'.format(input_path))
    logger.info('Input file: {}'.format(input_path))


    image = cv.imread(input_path)
    logger.info('Image was loaded from input file with shape: {}'.format(image.shape))


    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    logger.info('Input image has been converted to gray scale. Shape: {}'.format(gray_image.shape))

    contrast_gray_image = fix_contrast(gray_image)

    edges = cv.Canny(contrast_gray_image, 100, 200)
    logger.info('Canny operator was applied!')



    cv.imshow('original', image)
    cv.imshow('gray_image', gray_image)
    cv.imshow('contrast_gray_image', contrast_gray_image)
    cv.imshow('edges', edges)
    show_edges(contrast_gray_image, edges)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
