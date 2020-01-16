"""
@version: python3.7
@author:wenqsh
@contact:
@software: PyCharm
@file:
@title: mtcnn_02
@time: 2020/01/14 11:43
@result:
"""
'''
目标：检查jpg图片和json文件，
'''
# 导包
import os
import time
import json
import cv2
import torch

# 定义常量
JSON_DIR = r"F:\Dataset\mctnn_dataset\json1"
PIC_DIR = r"F:\Dataset\mctnn_dataset\jpg1"
PICTEST_SAVEDIR = r"F:\Dataset\mctnn_dataset\pictest"
PREDATALIST_SAVEFILE = "../saves/predatalist1.torch"
PREDATADICT_SAVEFILE = "../saves/predatadict1.torch"
BOX_COLOR = (0, 255, 0)
THICKNESS = 3
if __name__ == '__main__':
    print(1)
    a = 0.2
    b = f"{a:8.2f}"
    c = f"{b:20s}"
    print(c)
    datalist = []
    datadict = {}
    for i, json_filename in enumerate(os.listdir(JSON_DIR)):
        pic_name = json_filename.split('.')[0]
        pic_filename = f"{pic_name}.jpg"



        pic_file = os.path.join(PIC_DIR, pic_filename)
        json_file = os.path.join(JSON_DIR, json_filename)
        pic_savefile = os.path.join(PICTEST_SAVEDIR, pic_filename)
        # print(json_file)
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                load_dict = json.load(f)
                xy = load_dict['outputs']['object'][0]['bndbox']
                x1, y1 = xy['xmin'], xy['ymin']
                x2, y2 = xy['xmax'], xy['ymax']
                w, h = (x2 - x1), (y2 - y1)
                width, height = load_dict['size']['width'], load_dict['size']['height']
                # 可以根据情况修改
                jsondata = "{0} {1} {2} {3} {4} {5} {6} {7} {8}".format(pic_filename,
                                                                        x1, y1, x2, y2, w, h, width, height)
                '''
                数据处理'''

                datalist.append(jsondata)
                datadict[pic_filename] = [x1, y1, x2, y2, w, h, width, height]


                '''
                图片处理'''
                # img = cv2.imread(pic_file)
                # cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)
                # cv2.imwrite(pic_savefile, img)
                # cv2.imshow(pic_name, img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            except Exception as e:
                print("Error: ", e)
                print(pic_filename)
        '''间断测试'''
        # if i > 0:
        #     print(datalist)
        #     print(len(datalist))
        #     break
    torch.save(datalist, PREDATALIST_SAVEFILE)
    torch.save(datadict, PREDATADICT_SAVEFILE)
    datas = torch.load(PREDATALIST_SAVEFILE)
    print(len(datas))
    print(type(datas[0]))
    print(datas[0])
    print("**************")
    datas = torch.load(PREDATADICT_SAVEFILE)
    print(type(datas))
    print(len(datas))
    for k, v in datas.items():
        print(k)
        print(v)
        print(type(v[0]))
        break
    # print(datas[-5:])
    # for d in datas:
    #     print(d)

