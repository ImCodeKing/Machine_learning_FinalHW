import time

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import preprocess_input


class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape=[600, 600], train=True):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        # 训练时进行数据的随机增强(random=self.train)
        # 验证时不进行数据的随机增强
        image, the_box = self.get_data(self.annotation_lines[index], self.input_shape[0:2], random=self.train)
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        bndbox_data = np.zeros((len(the_box), 5))
        if len(the_box) > 0:
            bndbox_data[:len(the_box)] = the_box

        box = bndbox_data[:, :4]
        label = bndbox_data[:, -1]
        return image, box, label

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    # 训练阶段，对数据集进行一些基础变换，以增加训练集数量
    def get_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        image = Image.open(line[0])

        # 获得图像的高宽与目标高宽
        image_w, image_h = image.size
        h, w = input_shape

        # 获得真实框
        bndbox = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w / image_w, h / image_h)
            new_w = int(image_w * scale)
            new_h = int(image_h * scale)
            dx = (w - new_w) // 2
            dy = (h - new_h) // 2

            image = image.resize((new_w, new_h), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            data = np.array(new_image, np.float32)

            if len(bndbox) > 0:
                np.random.shuffle(bndbox)
                bndbox[:, [0, 2]] = bndbox[:, [0, 2]] * new_w / image_w + dx
                bndbox[:, [1, 3]] = bndbox[:, [1, 3]] * new_h / image_h + dy
                bndbox[:, 0:2][bndbox[:, 0:2] < 0] = 0
                bndbox[:, 2][bndbox[:, 2] > w] = w
                bndbox[:, 3][bndbox[:, 3] > h] = h
                box_w = bndbox[:, 2] - bndbox[:, 0]
                box_h = bndbox[:, 3] - bndbox[:, 1]
                bndbox = bndbox[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid bndbox

            return data, bndbox

        # 计算新的宽高比并且进行随机的扭曲
        new_ar = image_w / image_h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            new_h = int(scale * h)
            new_w = int(new_h * new_ar)
        else:
            new_w = int(scale * w)
            new_h = int(new_w / new_ar)
        image = image.resize((new_w, new_h), Image.BICUBIC)

        # 处理image多余的部分
        dx = int(self.rand(0, w - new_w))
        dy = int(self.rand(0, h - new_h))
        # 原始图像灰色背景
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 随机翻转
        flip_lr = self.rand() < .5
        flip_tb = self.rand() < .5
        if flip_lr:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_tb:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        data = np.array(image, np.uint8)

        # 将图像映射到HSV色域，随机变色，色相（hue）、饱和度（sat）和亮度（val）
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(data, cv2.COLOR_RGB2HSV))

        x = np.arange(0, 256, dtype=r.dtype)
        rand_hue = ((x * r[0]) % 180).astype(data.dtype)
        rand_sat = np.clip(x * r[1], 0, 255).astype(data.dtype)
        rand_val = np.clip(x * r[2], 0, 255).astype(data.dtype)

        data = cv2.merge((cv2.LUT(hue, rand_hue), cv2.LUT(sat, rand_sat), cv2.LUT(val, rand_val)))

        # 生成随机噪声
        noise = self.rand() < .5
        if noise:
            # 生成一个与图像大小相同的随机二值遮罩
            mask = np.random.rand(data.shape[0], data.shape[1]) > 0.1
            # 将遮罩应用于图像
            data[np.where(mask==False)] = 0

        data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)

        # 根据之前的随机对先验框进行调整
        if len(bndbox) > 0:
            np.random.shuffle(bndbox)
            bndbox[:, [0, 2]] = bndbox[:, [0, 2]] * new_w / image_w + dx
            bndbox[:, [1, 3]] = bndbox[:, [1, 3]] * new_h / image_h + dy

            if flip_lr:
                bndbox[:, [0, 2]] = w - bndbox[:, [2, 0]]
            if flip_tb:
                bndbox[:, [1, 3]] = h - bndbox[:, [1, 3]]

            bndbox[:, 0:2][bndbox[:, 0:2] < 0] = 0
            bndbox[:, 2][bndbox[:, 2] > w] = w
            bndbox[:, 3][bndbox[:, 3] > h] = h
            box_w = bndbox[:, 2] - bndbox[:, 0]
            box_h = bndbox[:, 3] - bndbox[:, 1]
            bndbox = bndbox[np.logical_and(box_w > 1, box_h > 1)]

        return data, bndbox


# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = torch.from_numpy(np.array(images))
    return images, bboxes, labels
