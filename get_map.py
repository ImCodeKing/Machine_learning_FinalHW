import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_map
from frcnn import FRCNN

if __name__ == "__main__":
    classes_path = 'model_data/voc_classes.txt'
    VOCdevkit_path = 'VOCdevkit/'
    map_out_path = 'map_out'

    confidence = 0.02
    nms_iou = 0.5
    score_threhold = 0.5

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/val.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    frcnn = FRCNN(confidence=confidence, nms_iou=nms_iou)

    for image_id in tqdm(image_ids):
        image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
        image = Image.open(image_path)
        frcnn.get_map_txt(image_id, image, class_names, map_out_path)

    # 获取真实框
    for image_id in tqdm(image_ids):
        with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
            root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/" + image_id + ".xml")).getroot()
            for obj in root.findall('object'):
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text

                new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

    get_map(0.5, True, score_threhold=score_threhold, path=map_out_path)

