from PIL import Image

from frcnn import FRCNN

import os
from tqdm import tqdm

if __name__ == "__main__":
    frcnn = FRCNN()

    # dir_origin_path 指定了用于检测的图片的文件夹路径
    # dir_save_path 指定了检测完图片的保存路径
    dir_origin_path = "validation/"
    dir_save_path = "img_out/"

    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_origin_path, img_name)
            image = Image.open(image_path)

            r_image = frcnn.detect_image(image)

            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

