import torch
from torchvision import transforms
import time
import numpy as np
# from src import cfg
from tool import detectutils
from tool import utils
import cv2
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import os
from nets import PNet, RNet, ONet, Net

PNET_TRAINED = r"../params/pnet_07_4.pt"
RNET_TRAINED = r"../params/rnet_07_4.pt"
ONET_TRAINED = r"../params/onet_07_4.pt"
SAVE_DIR = r"../saves/image"
utils.makedir(SAVE_DIR)


class Detector():
    def __init__(self,returnnet="onet", pnet_f=PNET_TRAINED, rnet_f=RNET_TRAINED, onet_f=ONET_TRAINED,
                 trainnet:Net=None, isCuda=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 先尝试加载参数，再尝试加载模型
        try:
            self.pnet = PNet()
            self.rnet = RNet()
            self.onet = ONet()
            self.pnet.load_state_dict(torch.load(pnet_f, map_location='cuda'))
            self.rnet.load_state_dict(torch.load(rnet_f, map_location='cuda'))
            self.onet.load_state_dict(torch.load(onet_f, map_location='cuda'))
            print("netdict load successful")
        except:
            self.pnet = torch.load(pnet_f, map_location='cuda')
            self.rnet = torch.load(rnet_f, map_location='cuda')
            self.onet = torch.load(onet_f, map_location='cuda')
            print("net load successful")

        if trainnet != None:
            # {"pnet": self.pnet, "rnet": self.rnet, "onet": self.onet}[trainnet.name] = trainnet
            if trainnet.name == self.pnet.name:
                self.pnet = trainnet
            elif trainnet.name == self.rnet.name:
                self.rnet = trainnet
            elif trainnet.name == self.onet.name:
                self.onet = trainnet
            print("trainnet init finished", trainnet.mark)

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.THREHOLD = {
            12: [0.9, 0.3],
            24: [0.99, 0.3],
            48: [0.99, 0.3]
        }

        self.isCuda = isCuda

        if self.isCuda:
            self.pnet.to(self.device)
            self.rnet.to(self.device)
            self.onet.to(self.device)

        self.returnnet = returnnet

    def __makedir(self, path):
        '''
        如果文件夹不存在，就创建
        :param path:路径
        :return:路径名
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def detect(self, img_file):
        start_time = time.time()
        image = Image.open(img_file)

        pnet_boxes = self.__pnet_detect(image, self.THREHOLD[12][0], self.THREHOLD[12][1])

        if pnet_boxes.size(0) == 0:
            return np.array([])
        end_time = time.time()
        time_pnet = end_time - start_time

        start_time = time.time()
        rnet_boxes = self.__rnet_detect(pnet_boxes, image, self.THREHOLD[24][0], self.THREHOLD[24][1])
        if rnet_boxes.size(0) == 0:
            return np.array([])
        end_time = time.time()
        time_rnet = end_time - start_time

        start_time = time.time()
        onet_boxes = self.__onet_detect(rnet_boxes, image, self.THREHOLD[48][0], self.THREHOLD[48][1])
        if onet_boxes.size(0) == 0:
            return np.array([])
        end_time = time.time()
        time_onet = end_time - start_time
        time_total = time_pnet + time_rnet + time_onet

        print("totaltime: %.2f, pnet: %.2f, rnet: %.2f, onet: %.2f" % (time_total, time_pnet, time_rnet, time_onet))
        if self.returnnet == "pnet":
            return pnet_boxes.numpy()
        elif self.returnnet == "rnet":
            return rnet_boxes.numpy()
        elif self.returnnet == "onet":
            return onet_boxes.numpy()  # 返回numpy的原因是图像处理的需要
        else:
            raise ValueError

    def __box(self, cls_mask, cls, offset, scale, stride=2, side_len=12):
        '''

        :param index: 根据置信度筛选出来数据的索引，维度为2，0维代表行，一维代表列
        :param cls: 置信度数据
        :param offset: 偏移量数据
        :param scale: 缩放比例
        :param stride: 组合卷积步长
        :param side_len: 组合卷积卷积核大小
        :return:
        '''

        index = torch.nonzero(cls_mask).float()  # 这个转float很重要
        # 取置信数据
        confidence = cls[cls_mask]

        # 回原图
        # 原图四个解的坐标

        _x1 = index[:, 1] * stride / scale
        _y1 = index[:, 0] * stride / scale
        _x2 = (index[:, 1] * stride + side_len) / scale
        _y2 = (index[:, 0] * stride + side_len) / scale

        # 取偏移量的数据

        '''
        offset torch.Size([108, 196, 4]) torch.Size([108, 196])
        torch.Size([621]) torch.Size([4])'''
        _offset = offset.permute(1, 2, 0)[cls_mask]

        #  计算建议框大小
        _side = side_len // scale

        # 计算实际框四个坐标
        x1 = _x1 + _offset[:, 0] * _side
        y1 = _y1 + _offset[:, 1] * _side
        x2 = _x2 + _offset[:, 2] * _side
        y2 = _y2 + _offset[:, 3] * _side

        boxes = torch.stack((x1, y1, x2, y2, confidence), dim=1)
        torch.round_(boxes[:, :4])

        return boxes

    def __pnet_detect(self, image, clsthrehold, nmsthrehold):
        # 定义一个列表，放框
        boxes = []

        w, h = image.size
        min_side_len = min(w, h)  # 做图像金字塔需要最小边
        scale = 1.0  # 图像金字塔的缩放比例
        img = image
        while min_side_len > 12:  # 最小边大于12时做图像金字塔

            # 定义一个列表，放每一次缩放的框，用于单张特征图的nms
            # 进网前数据处理

            img_data = self.transform(img)  # 可以不经过numpy转换

            if self.isCuda:  # 设备
                img_data = img_data.to(self.device)

            img_data.unsqueeze_(0)  # 升维，下划线代表原位操作
            # 读取网络数据
            _cls, _offset = self.pnet(img_data)
            # 取出置信的单张特征图和偏移量的四张特征图
            cls, offset = _cls[0][0].detach().cpu(), _offset[0].detach().cpu()  # 出网络后转到cpu上算，并降维度
            # 置信条件的掩码

            cls_mask = torch.gt(cls, clsthrehold)

            if torch.any(cls_mask):
                # 返回满足置信条件的boxes，并做nms

                boxes.extend(utils.nms(self.__box(cls_mask, cls, offset, scale), nmsthrehold))

            # 改变scale
            scale = scale * 0.7

            # 图片resize
            _w, _h = int(w * scale), int(h * scale)
            img = img.resize((_w, _h))
            # 计算循环变量-新图片的最小边长
            min_side_len = min(_w, _h)

        return utils.nms(torch.stack(boxes), nmsthrehold)

    def __rnet_detect(self, pnet_boxes, image, clsthrehold, nmsthrehold):
        '''

        :param pnet_boxes:  p网络输出的box
        :param image:  图片
        :param clsthrehold: 置信度阈值
        :param nmsthrehold:  nms阈值
        :return:
        '''
        # 定义用于放实际框的列表
        _img_dataset = []
        img = image
        # 转成方形框
        _pnet_boxes = detectutils.to_square(pnet_boxes, img)

        for _box in _pnet_boxes:
            _img_dataset.append(self.transform(np.array(img.crop(_box.numpy()).resize((24, 24)))))
        if not _img_dataset:
            return torch.Tensor([])
        img_dataset = torch.stack(_img_dataset)  # list数据为tensor就可以用stack
        if self.isCuda:
            img_dataset = img_dataset.to(self.device)

        _cls, _offset = self.rnet(img_dataset)

        _cls = _cls.detach().cpu()
        _offset = _offset.detach().cpu()

        cls_mask = torch.gt(_cls, clsthrehold).view(-1)
        if torch.any(cls_mask):
            cls = _cls[cls_mask]
            offset = _offset[cls_mask]
            pnet_boxes_ = _pnet_boxes[cls_mask]
            # 正方形四个点的坐标
            _x1 = pnet_boxes_[:, 0]
            _y1 = pnet_boxes_[:, 1]
            _x2 = pnet_boxes_[:, 2]
            _y2 = pnet_boxes_[:, 3]
            # 根据偏移量计算新框坐标

            _side = _x2 - _x1
            _side1 = _y2 - _y1

            x1 = _x1 + offset[:, 0] * _side
            y1 = _y1 + offset[:, 1] * _side
            x2 = _x2 + offset[:, 2] * _side
            y2 = _y2 + offset[:, 3] * _side

            # boxes = torch.stack((x1, y1, x2, y2, cls.view(-1)), dim=1)
            boxes = torch.stack((x1, y1, x2, y2, cls.view(-1)), dim=1)

            torch.round_(boxes[:, :4])

            return utils.nms(boxes, nmsthrehold)
        return torch.Tensor([])

    def __onet_detect(self, rnet_boxes, image, clsthrehold, nmsthrehold):
        '''
        :param rnet_boxes:  p网络输出的box
        :param image:  图片
        :param clsthrehold: 置信度阈值
        :param nmsthrehold:  nms阈值
        :return:
        '''
        # 定义用于放实际框的列表
        img = image
        _img_dataset = []
        _rnet_boxes = detectutils.to_square(rnet_boxes, img)

        for _box in _rnet_boxes:
            _img_dataset.append(self.transform(np.array(img.crop(_box.numpy()).resize((48, 48)))))
        if not _img_dataset:
            return torch.Tensor([])
        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.to(self.device)
        _cls, _offset = self.onet(img_dataset)

        _cls = _cls.detach().cpu()
        _offset = _offset.detach().cpu()

        cls_mask = torch.gt(_cls, clsthrehold).view(-1)

        if torch.any(cls_mask):
            cls = _cls[cls_mask]
            # print("cls",cls)
            offset = _offset[cls_mask]
            rnet_boxes_ = _rnet_boxes[cls_mask]
            # 正方形四个点的坐标
            _x1 = rnet_boxes_[:, 0]
            _y1 = rnet_boxes_[:, 1]
            _x2 = rnet_boxes_[:, 2]
            _y2 = rnet_boxes_[:, 3]
            # 根据偏移量计算新框坐标
            _side = _x2 - _x1
            x1 = _x1 + offset[:, 0] * _side
            y1 = _y1 + offset[:, 1] * _side
            x2 = _x2 + offset[:, 2] * _side
            y2 = _y2 + offset[:, 3] * _side

            boxes = torch.stack((x1, y1, x2, y2, cls.view(-1)), dim=1)
            torch.round_(boxes[:, :4])

            rboxes = utils.nms(boxes, nmsthrehold, isMin=True)
            return rboxes
        return torch.Tensor([])

    def image_genarate(self, image_path, savedir=None, img_name=None, show=False, color=(0, 255, 0), thickness=2):
        '''

        :param image_path:  图片路径
        :param savedir:  要保存的文件夹，如果为None，就不保存
        :param show:  是否展示，默认False
        :param color: box颜色，默认(0,255,0)
        :param thickness: box宽度，默认2
        :return:
        '''

        img = cv2.imread(image_path)
        boxes = self.detect(image_path)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            pt1 = (x1, y1)
            pt2 = (x2, y2)
            cv2.rectangle(img, pt1, pt2, color, thickness)
        if img_name == None:
            img_name = os.path.basename(image_path)
        if savedir:
            self.__makedir(savedir)
            img_file = os.path.join(savedir, img_name)
            cv2.imwrite(img_file, img)
        if show or not savedir:
            cv2.imshow(img_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf, suppress=False)
    torch.set_printoptions(threshold=np.inf, sci_mode=False)

    img_file = r"../images/012.jpg"
    save_dir = r"../saves/image"
    detecter = Detector()
    flag = True
    detecter.image_genarate(img_file, save_dir if flag else None, "a.jpg")
