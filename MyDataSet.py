# -*- coding: utf-8 -*-
# @Time    : 2018/6/13 15:01
# @Author  : zhoujun
import cv2
import numpy as np
import torch.utils.data as Data
import os
from PIL import Image
from typing import List
from natsort import natsorted
from torchvision import transforms
import matplotlib.pyplot as plt
import pathlib

def get_file_list(folder_path: str, p_postfix: str or list = ['.jpg'], sub_dir: bool = True) -> list:
    """
    获取所给文件目录里的指定后缀的文件,读取文件列表目前使用的是 os.walk 和 os.listdir ，这两个目前比 pathlib 快很多
    :param filder_path: 文件夹名称
    :param p_postfix: 文件后缀,如果为 [.*]将返回全部文件
    :param sub_dir: 是否搜索子文件夹
    :return: 获取到的指定类型的文件列表
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    if isinstance(p_postfix,str):
        p_postfix = [p_postfix]
    file_list = []
    if sub_dir:
        for rootdir, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(rootdir, file)
                for p in p_postfix:
                    if os.path.isfile(file_path) and (file_path.endswith(p) or p == '.*'):
                        file_list.append(file_path)
    else:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            for p in p_postfix:
                if os.path.isfile(file_path) and (file_path.endswith(p) or p == '.*'):
                    file_list.append(file_path)
    return natsorted(file_list)


class MyDataSet(Data.Dataset):
    def __init__(self, txt, data_shape, channel=3, transform=None, target_transform=None):
        '''
        :param txt: 存放图片和标签的文本，其中数据和标签以空格分隔，一行代表一个样本
        :param data_shape: 图片的输入大小
        :param channel: 图片的通道数
        :param transform: 数据的tarnsform
        :param target_transform: 标签的target_transform
        '''
        with open(txt, 'r') as f:
            data = list(line.strip().split(' ') for line in f if line)
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.data_shape = data_shape
        self.channel = channel

    def __readimg__(self, img_path, transform):
        img = cv2.imread(img_path, 0 if self.channel == 1 else 3)
        img = cv2.resize(img, (self.data_shape[0], self.data_shape[1]))
        img = np.reshape(img, (self.data_shape[0], self.data_shape[1], self.channel))
        if transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        img_path, label_path = self.data[index]
        return self.__readimg__(img_path, self.transform), self.__readimg__(img_path, self.target_transform)

    def __len__(self):
        return len(self.data)


class ImageData(Data.Dataset):
    def __init__(self, img_root, transform=None, t_transform=None):
        self.image_path = get_file_list(img_root,p_postfix=['.jpg'],sub_dir=True)
        self.image_path = [x for x in self.image_path if pathlib.Path(x).stat().st_size > 0]

        self.label_path = [x + '.npy' for x in self.image_path]
        # self.label_path = [x.replace('add_bg_img_2000_1180_nnn', 'src_img') for x in self.image_path]
        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, index):
        # image = Image.open(self.image_path[index])
        # label = Image.open(self.label_path[index])
        # label = label.resize(image.size)
        image = cv2.imread(self.image_path[index])
        label = np.load(self.label_path[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        return image, label

    def __len__(self):
        return len(self.image_path)


if __name__ == '__main__':
    img_path = 'Z:/zj/data/add_bg_img_2000_1180_nnn'
    test_data = ImageData(img_path, transform=transforms.ToTensor(), t_transform=transforms.ToTensor())
    test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=3)
    import torch
    loss_fn = torch.nn.MSELoss()
    for img, label in test_loader:
        #     print(img[0].permute(1,2,0).numpy().shape)
        #     print(label.shape)
        #     print(img.dtype)
        #     print(img.shape)
        loss = loss_fn(img,label)
        show_img = img[0].permute(1, 2, 0).numpy()
        plt.imshow(show_img)
        plt.show()
        label = label[0].permute(1, 2, 0).numpy()
        label_img = cv2.remap(show_img, label[:, :, 0], label[:, :, 1], cv2.INTER_LINEAR)
        plt.imshow(label_img)
        plt.show()
        break
