# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 11:30 上午
# @Author  : Haonan Wang
# @File    : Load_Dataset.py
# @Software: PyCharm
import numpy as np
import torch
import random
from PIL import Image
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Any, Callable, List, Optional, Sequence, Tuple
from collections import OrderedDict
import os
import cv2
from scipy import ndimage


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = F.to_pil_image(image), F.to_pil_image(label)

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)


        image = F.to_tensor(image)
        label = to_long_tensor(label)
        sample = {'image': image, 'label': label}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = F.to_pil_image(image), F.to_pil_image(label)

        image = F.to_tensor(image)
        label = to_long_tensor(label)
        sample = {'image': image, 'label': label}
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class ImageToImage2D(Dataset):
    

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask
        

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))   #返回数据集中图像的数量

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]
        
        image = Image.open(os.path.join(self.input_path, image_filename)).convert('RGB')  #这是添加的图像，抽取的图像
        
        image = np.array(image.resize((self.image_size, self.image_size),Image.BICUBIC))  #进行缩放操作
        
        mask =Image.open(os.path.join(self.output_path, image_filename)).convert('L') #灰度图和rgb图的转变
        
        mask = np.array(mask.resize((self.image_size, self.image_size),Image.NEAREST))
        # print(mask.shape)#
        # print(2)
        mask[mask <= 0] = 0
        
        mask[mask > 0] = 1
        

       
        image, mask = correct_dims(image, mask)

        
        
        sample = {'image': image, 'label': mask}

        if self.joint_transform:
            sample = self.joint_transform(sample)
        


        

        return sample, image_filename
