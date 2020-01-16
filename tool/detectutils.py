# from src import *
from tool.utils import *
def to_square(boxes_, image):
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