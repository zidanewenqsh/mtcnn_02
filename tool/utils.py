import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import time

def formattime(start_ms, end_ms):
    ms = end_ms - start_ms
    m_end, s_end = divmod(ms, 60)
    h_end, m_end = divmod(m_end, 60)
    time_data = "%02d:%02d:%02d" % (h_end, m_end, s_end)
    return time_data

def makedir(path):
    '''
    如果文件夹不存在，就创建
    :param path:路径
    :return:路径名
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def toTensor(data):
    '''

    :param data:
    :return:
    '''
    if isinstance(data, torch.FloatTensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    elif isinstance(data, (list, tuple)):
        return torch.tensor(list(data)).float()  # 针对列表和元组，注意避免list里是tensor的情况
    elif isinstance(data, torch.Tensor):
        return data.float()
    return


def toNumpy(data):
    '''

    :param data:
    :return:
    '''
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, (list, tuple)):
        return np.array(list(data))  # 针对列表和元组
    return

def toList(data):
    '''

    :param data:
    :return:
    '''
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, torch.Tensor):
        return data.numpy().tolist()
    elif isinstance(data, (list, tuple)):
        return list(data)  # 针对列表和元组
    return

def isBox(box):
    '''
    判断是否是box
    :param box:
    :return:
    '''
    box = toNumpy(box)
    if box.ndim == 1 and box.shape == (4,) and np.less(box[0], box[2]) and np.less(box[1], box[3]):
        return True
    return False

def isBoxes(boxes):
    '''
    判断是否是boxes
    :param boxes:
    :return:
    '''
    boxes = toNumpy(boxes)
    if boxes.ndim == 2 and boxes.shape[1] == 4:
        if np.less(boxes[:, 0], boxes[:, 2]).all() and np.less(boxes[:, 1], boxes[:, 3]).all():
            return True
    return False


def area(box):
    return torch.mul((box[2] - box[0]), (box[3] - box[1]))


def areas(boxes):
    return torch.mul((boxes[:, 2] - boxes[:, 0]), (boxes[:, 3] - boxes[:, 1]))

# iou
def iou(box, boxes, isMin=False):
    '''

    :param box:
    :param boxes:
    :param isMin:
    :return:
    '''
    '''
    define iou function
    '''

    box = toTensor(box)

    boxes = toTensor(boxes)  # 注意boxes为二维数组

    # 如果boxes为一维，升维
    if boxes.ndimension() == 1:
        boxes = torch.unsqueeze(boxes, dim=0)

    # box_area = torch.mul((box[2] - box[0]), (box[3] - box[1]))  # the area of the first row
    # boxes_area = torch.mul((boxes[:, 2] - boxes[:, 0]), (boxes[:, 3] - boxes[:, 1]))  # the area of other row

    box_area = area(box)
    boxes_area = areas(boxes)
    xx1 = torch.max(box[0], boxes[:, 0])
    yy1 = torch.max(box[1], boxes[:, 1])
    xx2 = torch.min(box[2], boxes[:, 2])
    yy2 = torch.min(box[3], boxes[:, 3])

    inter = torch.mul(torch.max((xx2 - xx1), torch.Tensor([0, ])), torch.max((yy2 - yy1), torch.Tensor([0, ])))
    # print("inter",inter.shape, box_area.shape, boxes_area.shape, box_area)

    if (isMin == True):
        over = torch.div(inter, torch.min(box_area, boxes_area))  # intersection divided by union
    else:
        over = torch.div(inter, (box_area + boxes_area - inter))  # intersection divided by union
    return over


def nms(boxes_input, threhold=0.3, isMin=False):
    '''
    define nms function
    :param boxes_input:
    :param isMin:
    :param threhold:
    :return:
    '''
    # print("aaa",boxes_input[:,:4].shape)
    if isBoxes(boxes_input[:, :4]):
        '''split Tensor'''
        boxes = toTensor(boxes_input)

        boxes = boxes[torch.argsort(-boxes[:, 4])]

        r_box = []
        while (boxes.size(0) > 1):
            r_box.append(boxes[0])
            mask = torch.lt(iou(boxes[0], boxes[1:], isMin), threhold)
            boxes = boxes[1:][mask]  # the other row of Tensor
            '''mask 不能直接放进来,会报IndexError'''
        if (boxes.size(0) > 0):
            r_box.append(boxes[0])
        if r_box:
            return torch.stack(r_box)  # 绝对不能转整数，要不然置信度就变成0
    elif isBox(boxes_input):
        return toTensor(boxes_input)
    return torch.Tensor([])
    # return torch.stack(r_box).long()

def to_square1(boxes_, image):
    boxes = boxes_.clone()
    # 图像的大小
    width, height = image.size
    # 框的大小
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    # 中心点
    cx = (boxes[:, 2] + boxes[:, 0]) // 2
    cy = (boxes[:, 3] + boxes[:, 1]) // 2
    # 方框的大小
    side_len = torch.max(w, h)
    # 方框的两点坐标
    _x1 = cx - side_len // 2
    _x2 = _x1 + side_len
    _y1 = cy - side_len // 2
    _y2 = _y1 + side_len
    # 边缘数据的掩码
    maskx1 = torch.lt(_x1, 0)
    masky1 = torch.lt(_y1, 0)
    maskx2 = torch.gt(_x2, width)
    masky2 = torch.gt(_y2, height)
    # 边缘数据处理
    _x2[maskx1] -= _x1[maskx1]
    _x1[maskx1] = 0
    _x1[maskx2] -= (_x2[maskx2] - width)
    _x2[maskx2] = width

    _y2[masky1] -= _y1[masky1]
    _y1[masky1] = 0
    _y1[masky2] -= (_y2[masky2] - height)
    _y2[masky2] = height

    return torch.stack((_x1, _y1, _x2, _y2), dim=1).float()

def drawrectangle(imgfile, box, outline='red', width=1, show=False):
    '''
    画矩阵框
    :param imgfile:
    :param box:
    :param outline:
    :param width:
    :return:
    '''
    box = toNumpy(box)
    if isBoxes(box):
        for b in box:
            drawrectangle(imgfile, b, outline, width)
    elif isBox(box):
        with Image.open(imgfile) as img:
            draw = ImageDraw.Draw(img)
            draw.rectangle(box.tolist(), outline=outline, width=width)
            if show:
                img.show()
    return

def imgTreat(picfile, savefile, box, size):
    '''
    图片处理：crop, resize, save
    box可接收一维或二维的数据
    :param picfile:
    :param savefile:
    :param box:
    :param size:
    :return:
    '''
    box = toNumpy(box)
    size = size if isinstance(size, tuple) else (size, size)
    if isBoxes(box):
        box = box[0]
    if isBox(box):
        with Image.open(picfile) as img:
            if img.mode == "RGB":
                img = img.crop(box)
                img = img.resize(size)
                img.save(savefile)
                return True
    return False

def getoffset(data, wh_scale, size_scale):
    '''

    :param data:
    :param wh_scale:中心点偏移比例
    :param size_scale:大小缩放比例
    :return:
    '''

    # 实际框 x1, y1, x2, y2, w, h, width, height = map(int, data[1:9])
    name, x1, y1, x2, y2, w, h, width, height = [int(x) if x.isdigit() else x for x in data]
    # 生成中心点坐标
    cx, cy = x1 + w // 2, y1 + h // 2

    # 生成中心点偏移
    # 中心点偏移量
    w0, h0 = map(int, [w * wh_scale, h * wh_scale])
    # 偏移中心点坐标

    cx_ = cx + np.random.randint(-w0, w0)
    cy_ = cy + np.random.randint(-h0, h0)


    # 生成偏移框大小,人脸成正方形
    if size_scale > 1:
        size_scale = 1.0 / size_scale

    side_len = np.random.randint(int(min(w, h) * size_scale), np.ceil(max(w, h) * (1.0 / size_scale)))

    x1_ = cx_ - side_len // 2
    y1_ = cy_ - side_len // 2
    x2_ = x1_ + side_len
    y2_ = y1_ + side_len

    # 计算偏移量
    offset_x1 = (x1 - x1_) / side_len
    offset_y1 = (y1 - y1_) / side_len
    offset_x2 = (x2 - x2_) / side_len
    offset_y2 = (y2 - y2_) / side_len

    box = [x1, y1, x2, y2]
    box_ = [x1_, y1_, x2_, y2_]  # boxes注意要升维,已在iou方法里升维

    return name, [offset_x1, offset_y1, offset_x2, offset_y2], box, box_

def checkoffset(offset, box, box_):
    box = toNumpy(box)
    box_ = toNumpy(box_)
    # x1, y1, x2, y2 = map(lambda x: x, box)
    # x1_, y1_, x2_, y2_ = map(lambda x: x, box_)
    side_len = box_[3] - box_[1]
    # _x1, _y1, _x2, _y2 = max(lambda x: x, np.subtract(box_, box))

    offset_ = np.divide(np.subtract(box, box_), side_len)
    return np.subtract(offset, offset_)


def offsetToBox(offset, box_):
    box_ = toNumpy(box_)
    offset = toNumpy(offset)
    side_len = box_[3] - box_[1]
    box = box_ + offset * side_len
    return np.round_(box)

def offsetToCord(offset, _side):
    x1 = offset[0] * _side
    y1 = offset[1] * _side
    x2 = _side + offset[2] * _side
    y2 = _side + offset[3] * _side
    box = [int(x) for x in [x1, y1, x2, y2]]
    return box

if __name__ == '__main__':
    torch.manual_seed(100)
    a = np.array([24, 23, 55, 66])
    b = torch.from_numpy(a).float()
    bs = torch.Tensor([[2, 2, 30, 30, 40], [3, 3, 25, 25, 60], [18, 18, 27, 27, 15]])
    t1 = time.time()
    for i in range(1000):
        x = iou(a, bs)
        y = nms(bs)
        # print("iou",utils_c.iou(a, bs))
        # print("nms", utils_c.nms(bs))
    t2 = time.time()
    print(t2 - t1)
    # print("iou",iou(a, bs))
    # print(nms(bs))
#     box = [1, 1, 4, 4]
#     box = (1, 2, 3, 4)
#     print(isinstance(box, tuple))
#     print(torch.Tensor(box))
#     boxes = bs[:, :4]
#     print(area(box))
#     print(areas(boxes))
#     '''
#     box: [129, 133, 318, 334]
# boxes [97, 169, 299, 326]'''
#     box = [129, 133, 318, 334]
#     boxes = [97, 169, 299, 326]
#     print(iou(box, boxes))

#     data01 = "000001.jpg  141   85  301  334  160  249    409    687"
#     data02 = data01.strip().split()
#     print(len(data02))
#     print(data02)
#     # print(getoffset(data02, 0.2, 0.8))
#     name, offset, box, box_ = getoffset(data02, 0.2, 0.8)
#     print(checkoffset(offset, box, box_))
#     box1 = np.array(box)
#     box1_ = np.array(box_)
#     print(box1.shape, box1_.shape, np.array(offset).shape)
#     print("offset", offset)
#     print("box_", box_)
#     print("box", box)
#     print(offsetToBox(offset, box_))
#     print(not isBox(box))
#     a = torch.Tensor([[1, 3, 3, 3], [2, 3, 5, 6], [3, 4, 7, 8]])
#     print(isBoxes(a))
#     boxes = torch.Tensor([[31.0000, 54.0000, 240.0000, 241.0000, 0.5618],
#                           [91.0000, 40.0000, 281.0000, 236.0000, 0.8155],
#                           [3.0000, 83.0000, 196.0000, 260.0000, 0.5850],
#                           [23.0000, 111.0000, 275.0000, 303.0000, 0.9364],
#                           [84.0000, 116.0000, 314.0000, 316.0000, 0.9743],
#                           [9.0000, 124.0000, 220.0000, 319.0000, 0.8091],
#                           [26.0000, 149.0000, 279.0000, 345.0000, 0.9504],
#                           [87.0000, 167.0000, 323.0000, 364.0000, 0.9123],
#                           [10.0000, 168.0000, 218.0000, 372.0000, 0.7569],
#                           [42.0000, 177.0000, 264.0000, 369.0000, 0.8784],
#                           [101.0000, 185.0000, 296.0000, 373.0000, 0.6029],
#                           [11.0000, 228.0000, 252.0000, 429.0000, 0.5594],
#                           [70.0000, 220.0000, 274.0000, 412.0000, 0.8263],
#                           [-10.0000, 294.0000, 274.0000, 485.0000, 0.6613],
#                           [72.0000, 295.0000, 313.0000, 490.0000, 0.9254],
#                           [141.0000, 273.0000, 330.0000, 471.0000, 0.5299],
#                           [53.0000, 317.0000, 295.0000, 501.0000, 0.8993],
#                           [120.0000, 311.0000, 328.0000, 504.0000, 0.5373],
#                           [38.0000, 319.0000, 262.0000, 503.0000, 0.6784]])
#     print(isBoxes(boxes))
#     boxes1 = toNumpy(boxes)
#     print(boxes1.ndim, boxes1.shape)
#     print(nms(boxes1))
#
