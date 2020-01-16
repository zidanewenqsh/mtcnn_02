import torch
import os
from PIL import Image, ImageDraw
import numpy as np
label_path = r"F:\Dataset\mctnn_dataset\save_10261_20200114\label\label_48.torch"
# path1 = r"F:\Dataset\mctnn_dataset\save_10261_20200114\data\data_12.torch"
pic_dir = r"F:\Dataset\mctnn_dataset\save_10261_20200114\pic\48"
side_len = 48
a = torch.load(label_path)
# for picname in os.listdir(pic_dir):
#     pic_path = os.path.join(pic_dir, picname)
#     print(pic_path)
labeldict:dict = torch.load(label_path)
for k, v in labeldict.items():
    print(k)
    conf = k.split('_')[2]
    # print(conf)
    if conf == "2":
        offset = torch.Tensor(v[1:])*side_len
        offset = offset.long()
        print(offset)
        x1 = offset[0]
        y1 = offset[1]
        x2 = side_len+offset[2]
        y2 = side_len+offset[3]
        print(x1,y1,x2,y2)
        img_file = os.path.join(pic_dir, k)

        with Image.open(img_file) as img:
            draw = ImageDraw.Draw(img)
            draw.rectangle((x1,y1,x2,y2),outline=(0,255,255),width=2)
            img.show()
        # break