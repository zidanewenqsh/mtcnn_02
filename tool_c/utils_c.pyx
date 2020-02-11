# utils_c.pyx
import torch
import numpy as np
cimport numpy as np
cimport cython


cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b

# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=2] _iou(np.ndarray[np.float32_t, ndim=1] box,
                                           np.ndarray[np.float32_t, ndim=2] boxes, int mode=0):
    # cdef int box_area
    cdef np.ndarray[np.float32_t, ndim=1] box_area, boxes_areas, inter
    cdef xx1, xx2, yy1, yy2

    box_area = (box[2:3] - box[0:1]) * (box[3:4] - box[1:2])

    boxes_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    # xx1 = max(box[0], boxes[:, 0])
    # yy1 = max(box[1], boxes[:, 1])
    # xx2 = min(box[2], boxes[:, 2])
    # yy2 = min(box[3], boxes[:, 3])
    inter = (np.maximum((xx2 - xx1), 0)*np.maximum((yy2 - yy1), 0))
    # return inter/((box_area+boxes_areas)-inter)

    if mode == 0:
        return inter/((box_area+boxes_areas)-inter)
    elif mode == 1:
        return inter/np.minimum(box_area, boxes_areas)
    else:
        return None

cdef np.ndarray[np.float32_t, ndim=2] _nms(np.ndarray[np.float32_t, ndim=2] boxes, float thresh=0.3, int mode=0):
    keep_boxes = []
    cdef np.ndarray[np.float32_t, ndim=1] _box, _ious
    cdef np.ndarray[np.float32_t, ndim=2] sort_boxes, _boxes, keep_boxes_
    sort_boxes = boxes[np.argsort(-boxes[:, 4])]
    while sort_boxes.shape[0] > 0:
        # print(boxes.shape[0])
        _box = sort_boxes[0]

        keep_boxes.append(_box)
        if boxes.shape[0] > 1:
            _boxes = sort_boxes[1:]
            _ious = _iou(_box, _boxes)
            sort_boxes = _boxes[np.less(_ious, thresh)]
        else:
            break
    keep_boxes_ = np.stack(keep_boxes)
    return keep_boxes_

def iou(box, boxes, mode=0):
    return _iou(box, boxes, mode=0)


def nms(boxes, thresh=0.3, mode=0):
    return _nms(boxes, thresh=0.3, mode=0)