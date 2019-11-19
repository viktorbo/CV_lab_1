import os
import logging as log
from color_transformation import rgb2gray
from image_readers import read_img_as_arr
from datetime import datetime
from PIL import Image

log.basicConfig(level=log.DEBUG,

                format=('{} [%(levelname)s] %(message)s').format(str(datetime.now().strftime('%d-%m-%Y %H:%M:%S'))))

def main():
    logger = log.getLogger(__name__)

    img = read_img_as_arr('/home/viktorbo/localdata/datasets/flower_photos/tulips/489506904_9b68ba211c.jpg')
    img_gray = rgb2gray(img, 'mid')




if __name__ == "__main__":
    main()
