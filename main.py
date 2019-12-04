import os
import logging as log
from datetime import datetime
from pathlib import Path
from PIL import Image
import cv2 as cv
import argparse

log.basicConfig(level=log.DEBUG,
                format=('{} [%(levelname)s] %(message)s').format(str(datetime.now().strftime('%d-%m-%Y %H:%M:%S'))))

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





if __name__ == "__main__":
    main()
