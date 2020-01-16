#!/usr/bin/env python
# D:\MySoft\Anaconda3 python
# coding:UTF-8
"""
@version: python3.7
@author:wenqsh
@contact:
@software: PyCharm
@file:
@title: mtcnndataset
@time: 2019/09/09 16:43
@result:
"""
import os
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import time
import torch

class Dataset(data.Dataset):
    def __init__(self, label_dir, pic_dir, size):  # transform要有，因为图片需要transform
        super().__init__()
        # self.label_path = label_path
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = []

        self.picdict = {}
        self.labeldict = {}
        i = 0
        for picdirs in os.listdir(pic_dir):
            # if str(size) in picdirs:

            if picdirs.startswith(str(size)):
                self.picdict[i] = os.path.join(pic_dir, picdirs) # 每个文件夹下的图片起始数字不同
                i += 1
        # j = 0
        for labelname in os.listdir(label_dir):
            labelname_ = labelname.split('.')[0]
            if labelname_.endswith(str(size)):
                tempdict = torch.load(os.path.join(label_dir, labelname))
                self.labeldict.update(tempdict)
        self.dataset = list(self.labeldict.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        pic_filename = self.dataset[item]
        pic_subdir = self.picdict[int(pic_filename[0])]
        pic_file = os.path.join(pic_subdir, pic_filename)
        with Image.open(pic_file) as img:
            img_data = self.transforms(img)
            # print("img_data", img_data)
        offset_x1, offset_y1 = self.labeldict[pic_filename][1], self.labeldict[pic_filename][2]
        offset_x2, offset_y2 = self.labeldict[pic_filename][3], self.labeldict[pic_filename][4]
        conf = self.labeldict[pic_filename][0]
        offset_target = torch.from_numpy(np.array([offset_x1, offset_y1, offset_x2, offset_x2], dtype=np.float32))
        conf_target = torch.from_numpy(np.array([conf], dtype=np.float32))

        return img_data, conf_target, offset_target

if __name__ == '__main__':
    # 存放标签和图片的文件夹
    label_dir = r"F:\Dataset\mctnn_dataset\save_10261_20200114\label"
    pic_dir = r"F:\Dataset\mctnn_dataset\save_10261_20200114\pic"
    dataset_ = Dataset(label_dir, pic_dir, 48)
    dataloader = data.DataLoader(dataset_, batch_size=50, shuffle=True, num_workers=1, drop_last=True)
    print(len(dataloader))
    for i, (img_data_, confidence_, offset_) in enumerate(dataloader):
        print(img_data_.size())
        print(confidence_.size(), confidence_)
        print(offset_.size(), offset_)
        print("***********")
        if i>1:
            print(i)
            print(img_data_.shape)
            break
