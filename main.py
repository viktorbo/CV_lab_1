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
        logger.error('Wrong shape of image (image has {} channels instead of 1). Converter was not applied!'.format(
            img.shape[2]))
        return img

    tmp = copy(img)
    for i in range(h):
        for j in range(w):
            tmp[i][j] = (tmp[i][j] - min) * k

    return tmp


def show_edges(img: np.ndarray, edges, size: int = 1, show: bool = False):
    logger = log.getLogger('show_edges')
    logger.info('Run show_edges')

    try:
        h, w = img.shape
    except ValueError:
        logger.error('Wrong shape of image (image has {} channels instead of 1).'.format(img.shape[2]))
        return

    tmp = copy(img)
    r = size // 2
    for i in range(r, h - r):
        for j in range(r, w - r):
            if edges[i][j] != 0:
                for x in range(i - r, i + r + 1):
                    for y in range(j - r, j + r + 1):
                        tmp[x][y] = 0
    if show:
        cv.imshow('Edges', tmp)
    return tmp


def find_corners(img: np.ndarray, blocksize, aperturesize, free):
    logger = log.getLogger('find_corners')
    logger.info('Run find_corners')
    if blocksize is None and aperturesize is None and free is None:
        logger.warning('Harris corner detector uses default params')
        blocksize = 2
        aperturesize = 3
        free = 0.04
    else:
        if blocksize is None:
            blocksize = 2
            logger.warning('Harris corner detector uses default block size')
        else:
            blocksize = blocksize

        if aperturesize is None:
            aperturesize = 3
            logger.warning('Harris corner detector uses default aperture size')
        else:
            aperturesize = aperturesize

        if free is None:
            free = 0.04
            logger.warning('Harris corner detector uses default "free" param')
        else:
            free = free
    logger.info('Harris corner detector uses the following params:')
    logger.info('   Block size:    {}'.format(blocksize))
    logger.info('   Aperture size: {}'.format(aperturesize))
    logger.info('   Free param:    {}'.format(free))

    corners = cv.cornerHarris(img, blocksize, aperturesize, free)
    norm = np.empty(corners.shape, dtype=np.float32)
    cv.normalize(corners, norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    scaled = cv.convertScaleAbs(norm)
    return scaled


def show_corners(img: np.ndarray, map, thresh, radius, thickness, color):
    logger = log.getLogger('show_corners')
    logger.info('Run show_corners')
    if thresh is None and radius is None and thickness is None and color is None:
        logger.warning('Corners will be shown with default params')
        thresh = 90
        radius = 10
        thickness = 1
        color = 0
    else:
        if thresh is None:
            thresh = 90
            logger.warning('Corners will be shown with default threshold')
        else:
            thresh = thresh

        if radius is None:
            radius = 10
            logger.warning('Corners will be shown with default radius')
        else:
            radius = radius

        if thickness is None:
            thickness = 1
            logger.warning('Corners will be shown with default thickness')
        else:
            thickness = thickness

        if color is None:
            color = 0
            logger.warning('Corners will be shown with default color (black)')
        else:
            color = color

    logger.info('Corners will be shown with following params:')
    logger.info('   Threshold: {}'.format(thresh))
    logger.info('   Radius:    {}'.format(radius))
    logger.info('   Thickness: {}'.format(thickness))
    logger.info('   Color:     {}'.format(color))

    tmp = copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if int(map[i, j]) > thresh:
                cv.circle(tmp, (j, i), radius, (color), thickness)
    cv.imshow('Corners with threshold = {}, radius = {}, thickness = {}'.format(thresh, radius, thickness), tmp)


def main():
    logger = log.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Path to input image')
    parser.add_argument('-bs', '--blocksize', type=int,
                        help='Neighborhood size for finding corners with Harris detector')
    parser.add_argument('-aps', '--apsize', type=int,
                        help='Aperture parameter for the Sobel operator in Harris corner detector')
    parser.add_argument('-fp', '--freeparam', type=float,
                        help='Harris detector free parameter. See the OpenCV docs for more information')
    parser.add_argument('-thr', '--thresh', type=int, help='Threshold for show corners with Harris corner detector')
    parser.add_argument('-r', '--radius', type=int, help='Radius of circle around corners for show corners with Harris corner detector')
    parser.add_argument('-t', '--thickness', type=int,
                        help='Thickness of circles (-1 for fill the circles) for show corners  with Harris corner detector')
    parser.add_argument('-c', '--color', type=int,
                        help='Color of circles in range of [0, 255] (from black to white) with Harris corner detector')
    args = parser.parse_args()

    if args.input is None:
        default_file_name = 'pyramid.png'
        default_path = os.path.abspath(default_file_name)
        logger.warning('The default input file will be used')
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
    # The contrast was improved

    # 100 and 200 are the most universal threshold values, so there are no configuration keys on the command line
    Tx = 100
    Ty = 200
    logger.info('Thresholds for Canny operator (Tx, Ty):  ({}, {})'.format(Tx, Ty))
    edges = cv.Canny(contrast_gray_image, Tx, Ty)
    logger.info('Canny operator was applied!')

    corners_map = find_corners(contrast_gray_image, args.blocksize, args.apsize, args.freeparam)

    show_corners(show_edges(contrast_gray_image, edges), corners_map, args.thresh, args.radius, args.thickness,
                 args.color)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
