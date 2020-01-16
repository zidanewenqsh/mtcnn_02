# from src import *
from tool.utils import *


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
    # 偏移坐标

    cx_ = cx + np.random.randint(-w0, w0)
    cy_ = cy + np.random.randint(-h0, h0)
    # print("cx",cx_,cx)

    '''
    #这两个是生成随机框的
    # w_ =  np.random.randint(int(w*0.8), int(w*1.2))
    # h_ = np.random.randint(int(h * 0.8), int(h * 1.2))

    '''
    # 生成偏移框大小,人脸成正方形
    if size_scale > 1:
        size_scale = 1.0 / size_scale
    side_len = np.random.randint(int(min(w, h) * size_scale), np.ceil(max(w, h) * (1.0 / size_scale)))

    # 建议框，生成新框坐标(实际w_和h_可能会偏小，因为取整的关系，但这样可以保证中心点坐正确)

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
    side_len = box_[3] - box_[1]
    offset_ = np.divide(np.subtract(box, box_), side_len)
    return np.subtract(offset, offset_)


def offsetToBox(offset, box_):
    box_ = toNumpy(box_)
    offset = toNumpy(offset)
    side_len = box_[3] - box_[1]
    box = box_ + offset * side_len
    return np.round_(box)


def generateLabeldata(picname, confidence, offset):
    '''
    将图片名称，偏移量，和置信处理成一组字符数据
    :param picname: str
    :param offset: list
    :param confidence: int
    :return:
    '''
    if confidence == 0:
        offset = [0.0, 0.0, 0.0, 0.0]
    data = []
    data.append(picname)
    data.append(confidence)
    data.extend(offset)

    label = "%-18s %10f %9f %9f %9f %9f" % tuple([d for d in data])
    return label


def generateLabel(datalist_write, label_savefile, face_size):
    '''

    :param datalist_write:
    :param label_savefile:
    :param face_size: 正方形框的大小，在这里主要是为了命名
    :return:
    '''
    positive_num = sum(list(map(lambda x: float(x.strip().split()[1]) == 1, datalist_write)))
    negative_num = sum(list(map(lambda x: float(x.strip().split()[1]) == 0, datalist_write)))
    part_num = sum(list(map(lambda x: float(x.strip().split()[1]) == 2, datalist_write)))
    total_num = positive_num + negative_num + part_num

    title = "size:{0},positive_num:{1},negative_num:{2},part_num:{3},total_num:{4}".format(
        face_size, positive_num, negative_num, part_num, total_num)
    axis = "%-18s %10s %9s %9s %9s %9s" \
           % ("name", "confidence", "offset_x1", "offset_y1", "offset_x2", "offset_y2")
    # 排序

    with open(label_savefile, 'w') as f:
        print(title, file=f)
        print(axis, file=f)
        for data in sorted(datalist_write):
            print(data, file=f)
    return
