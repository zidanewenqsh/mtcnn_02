import torch
import numpy as np
import os
import time

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


def area(box):
    box = toNumpy(box)
    return np.multiply(np.subtract(box[3], box[1]), np.subtract(box[2], box[0]))


def areas(boxes):
    boxes = toNumpy(boxes)
    # print(boxes)
    return np.multiply(np.subtract(boxes[:, 3], boxes[:, 1]), np.subtract(boxes[:, 2], boxes[:, 0]))


def iou(box, boxes, mode='inter'):
    # box = toNumpy(box)
    # boxes = toNumpy(boxes)
    # if boxes.ndim == 1:
    #     boxes = np.expand_dims(boxes, axis=0)
    box_area = area(box)
    boxes_areas = areas(boxes)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    #重点错处
    # inter = np.multiply(np.subtract(xx2, xx1), np.subtract(yy2, yy1))
    inter = np.multiply(np.maximum((xx2-xx1), 0), np.maximum((yy2- yy1),0))
    # print("inter",inter)
    # print("union",np.subtract(np.add(box_area, boxes_areas), inter))
    # print("area",box_area,boxes_areas)
    if mode == 'inter':
        return np.divide(inter, np.subtract(np.add(box_area, boxes_areas), inter))
    elif mode == 'min':
        return np.divide(inter, np.minimum(box_area, boxes_areas))


def nms(boxes, thresh=0.3, mode='inter'):
    # boxes = toNumpy(boxes)
    # if boxes.ndim == 1:
    #     return np.expand_dims(boxes, axis=0)
    keep_boxes = []
    if boxes.shape[0] == 0:
        return keep_boxes
    sort_boxes = boxes[np.argsort(-boxes[:, 4])]
    # print(sort_boxes.shape)
    # print(sort_boxes[0])
    while sort_boxes.shape[0] > 0:
        # print(boxes.shape[0])
        _box = sort_boxes[0]
        keep_boxes.append(_box)
        if boxes.shape[0] > 1:
            _boxes = sort_boxes[1:]
            _iou = iou(_box, _boxes, mode)
            sort_boxes = _boxes[np.less(_iou, thresh)]
        else:
            break
    return keep_boxes


if __name__ == '__main__':
    bs = np.array([[2, 2, 30, 30, 40], [3, 3, 25, 25, 60], [18, 18, 27, 27, 15]], dtype='f4')
    boxes = np.array([[31.0000, 54.0000, 240.0000, 241.0000, 0.5618],
                      [91.0000, 40.0000, 281.0000, 236.0000, 0.8155],
                      [3.0000, 83.0000, 196.0000, 260.0000, 0.5850],
                      [23.0000, 111.0000, 275.0000, 303.0000, 0.9364],
                      [84.0000, 116.0000, 314.0000, 316.0000, 0.9743],
                      [9.0000, 124.0000, 220.0000, 319.0000, 0.8091],
                      [26.0000, 149.0000, 279.0000, 345.0000, 0.9504],
                      [87.0000, 167.0000, 323.0000, 364.0000, 0.9123],
                      [10.0000, 168.0000, 218.0000, 372.0000, 0.7569],
                      [42.0000, 177.0000, 264.0000, 369.0000, 0.8784],
                      [101.0000, 185.0000, 296.0000, 373.0000, 0.6029],
                      [11.0000, 228.0000, 252.0000, 429.0000, 0.5594],
                      [70.0000, 220.0000, 274.0000, 412.0000, 0.8263],
                      [-10.0000, 294.0000, 274.0000, 485.0000, 0.6613],
                      [72.0000, 295.0000, 313.0000, 490.0000, 0.9254],
                      [141.0000, 273.0000, 330.0000, 471.0000, 0.5299],
                      [53.0000, 317.0000, 295.0000, 501.0000, 0.8993],
                      [120.0000, 311.0000, 328.0000, 504.0000, 0.5373],
                      [38.0000, 319.0000, 262.0000, 503.0000, 0.6784]], dtype='f4')
    t1 = time.time()
    for i in range(200000):
        # x = utils_c.iou(a, bs)
        # y = nms(boxes)
        z = nms(bs)
        # print("iou",utils_c.iou(a, bs))
        # print("nms", utils_c.nms(bs))
    t2 = time.time()
    print(t2 - t1)