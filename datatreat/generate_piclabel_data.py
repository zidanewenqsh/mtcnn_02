import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from tool import utils
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 定义路径
# root = r"D:\dataset"

PREDATALIST_FILE = "../saves/predatalist1.torch"  # 读取的标签路径
PREDATADICT_FILE = "../saves/predatadict1.torch"  # 读取的标签路径
PIC_PATH = r"F:\Dataset\mctnn_dataset\jpg"  # 读取的图片路径
SAVE_PATH = r"F:\Dataset\mctnn_dataset\save_10261_20200114"  # 保存路径

utils.makedir(SAVE_PATH)

# 图片和标签的保存根目录
pic_savedir = os.path.join(SAVE_PATH, "pic")
label_savedir = os.path.join(SAVE_PATH, "label")
data_savedir = os.path.join(SAVE_PATH, "data")
"define iou threshold"
positive_range = [0.65, 1]
part_range = [0.4, 0.6]
negative_range = [0, 0.05]
size_scale = 0.9
wh_scales = [0.05, 0.1, 0.2, 0.75, 0.95]
face_sizes = [12, 24, 48]

utils.makedir(pic_savedir)
utils.makedir(label_savedir)
utils.makedir(data_savedir)

# define 每张图片新生成图片的数量
each_pic_num = 10  # 1--144


def getconfidence(iou_value):
    '''
    通过iou值计算样本分类(正/负/部分)
    :param iou_value:
    :return:
    '''

    if positive_range[0] < iou_value < positive_range[1]:
        confidence = 1
    elif part_range[0] < iou_value < part_range[1]:
        confidence = 2
    elif negative_range[0] < iou_value < negative_range[1]:
        confidence = 0
    else:
        return -1
    return confidence


def generatedataset(pass_used=False, treat_img=True, gen_label=True):
    pic_used_dict = {12: "12.txt", 24: "24.txt", 48: "48.txt"}

    # 新旧标签列表
    picinfodict = torch.load(PREDATADICT_FILE)  # list

    ioulist = []  # 用于统计

    # 定义三个样本的计数
    positive_num_ = 0
    negative_num_ = 0
    part_num_ = 0

    positive_num = 0
    negative_num = 0
    part_num = 0

    for face_size in face_sizes:
        # datalist_write = []  # 注意datalist清空位置，注意计数
        datadict = {}
        labeldict = {}
        "标签保存路径"
        label_savefile = os.path.join(label_savedir, "label_{0}.torch".format(str(face_size)))
        data_savefile = os.path.join(data_savedir, "data_{0}.torch".format(str(face_size)))

        '''
           将已生成的图片存起来，下次不再生成'''
        pic_used_txt = pic_used_dict.get(face_size)
        pic_used_list = []
        if pass_used:

            if os.path.exists(pic_used_txt):
                with open(pic_used_txt) as f:
                    for line in f.readlines():
                        if line[0].isdigit():
                            pic_used_list.append(line.strip().split()[0])
        elif os.path.exists(pic_used_txt):
            os.remove(pic_used_txt)

        "遍历标签数据"
        for i, (name, datas) in enumerate(picinfodict.items()):
            "分隔每行数据"

            if pass_used:
                if name in pic_used_list:
                    continue

            x1, y1, x2, y2, w, h, width, height = datas
            # print("datas", datas)
            "图片过滤"
            # if (x1 < 0 or y1 < 0 or w < 0 or h < 0 or max(w, h) < 40):
            #     continue
            "判断是否有小于0的元素，框的大小是否比40小，框是否在图片内"
            if np.any(np.array(datas) < 0) or min(w, h) < 40 or max(w, h) > min(width, height):
                # print("false data: ", data)
                # 输出错误数据
                continue

            for j in range(len(wh_scales)):
                k = 0
                while k < each_pic_num:
                    '''
                    name是加载图片的名称
                    pic_name 是新生成图片的名称'''

                    '''
                    获取偏移量'''
                    wh_scale = wh_scales[j]
                    # 实际框 x1, y1, x2, y2, w, h, width, height = map(int, data[1:9])
                    # x1, y1, x2, y2, w, h, width, height = datas
                    # 生成中心点坐标
                    cx, cy = x1 + w // 2, y1 + h // 2

                    # 生成中心点偏移
                    # 中心点偏移量
                    w0, h0 = map(int, [w * wh_scale, h * wh_scale])
                    # 偏移中心点坐标

                    cx_ = cx + np.random.randint(-w0, w0)
                    cy_ = cy + np.random.randint(-h0, h0)

                    # 生成偏移框大小,人脸成正方形
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

                    "原图片路径"
                    pic_file = os.path.join(PIC_PATH, name)

                    iou_value = utils.iou(box, box_)

                    '置信度'
                    confidence = getconfidence(iou_value)  # 通过iou判断置信度

                    # 生成图片文件名

                    pic_name = "%s_%d_%d_0%1d%02d.jpg" % (name.split('.')[0], face_size, confidence, j, k)
                    '图片保存路径'
                    if confidence == -1:
                        continue
                    else:
                        ioulist.append(iou_value)
                        k += 1

                    pic_savedir_ = os.path.join(pic_savedir, str(face_size))
                    utils.makedir(pic_savedir_)
                    pic_savefile = os.path.join(pic_savedir_, pic_name)
                    '处理图片'
                    if treat_img:
                        # utils.imgTreat(pic_file, pic_savefile, box_, face_size)
                        with Image.open(pic_file) as img:
                            # if img.mode == "RGB":
                            img = img.crop(box_)
                            img = img.resize((face_size, face_size))
                            img_data = transform(img)
                            img.save(pic_savefile)
                            # torch.save(img_data, pic_savefile)

                    '处理标签'
                    datadict[pic_name] = img_data
                    if confidence == 0:
                        # labeldict[pic_name] = torch.Tensor([confidence, 0.0, 0.0, 0.0, 0.0])
                        labeldict[pic_name] = [confidence, 0.0, 0.0, 0.0, 0.0]
                    else:
                        # labeldict[pic_name] = torch.Tensor([confidence, offset_x1, offset_y1, offset_x2, offset_y2])
                        labeldict[pic_name] = [confidence, offset_x1, offset_y1, offset_x2, offset_y2]

                    '三个样本计数'
                    if positive_range[0] < iou_value < positive_range[1]:
                        positive_num_ += 1
                    elif part_range[0] < iou_value < part_range[1]:
                        part_num_ += 1
                    elif negative_range[0] < iou_value < negative_range[1]:
                        negative_num_ += 1

                if i % 100 == 0 and j == 0:
                    total_num_ = positive_num_ + part_num_ + negative_num_
                    print("epoch", i, positive_num_, part_num_, negative_num_, total_num_)
                    if all([positive_num_ > 0, part_num_ > 0, negative_num_ > 0, total_num_ > 0]):
                        print("positive: %.3f, %.3f" % (positive_num_ / part_num_, positive_num_ / negative_num_))
                        print("ratio: %.3f, %.3f, %.3f" % (positive_num_ / total_num_, part_num_ / total_num_,
                                                           negative_num_ / total_num_))  # 临时查看一下数量
                    print("********************")

            if pass_used:
                with open(pic_used_txt, 'a') as pic_used_file:
                    print(name, file=pic_used_file)
        if gen_label:
            torch.save(datadict, data_savefile)
            torch.save(labeldict, label_savefile)

    # 三个样本计数
    for i in ioulist:
        if positive_range[0] < i < positive_range[1]:
            positive_num += 1
        elif part_range[0] < i < part_range[1]:
            part_num += 1
        elif negative_range[0] < i < negative_range[1]:
            negative_num += 1
    total_num = positive_num + part_num + negative_num
    print(positive_num, part_num, negative_num, total_num)


if __name__ == '__main__':
    # 清空一下save文件夹
    # preclear(savepath)
    generatedataset(pass_used=False, gen_label=True, treat_img=True)
    # generatedataset(pass_used=False, gen_label=True, treat_img=True)
    # generatedataset(pass_used=True, gen_label=True, treat_img=True)
    print(1)
