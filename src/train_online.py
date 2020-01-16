import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor, optim
from PIL import Image, ImageDraw
import argparse
import time
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt



curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# from nets import PNet, RNet, ONet, Net

# from detect.mtcnndetect import Detector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "/content/drive/My Drive/mtcnn/saves/20200115"
PIC_DIR = "/content/drive/My Drive/mtcnn/datas/pic"
LABEL_DIR = "/content/drive/My Drive/mtcnn/datas/label"
NETFILE_EXTENTION = "pt"
ALPHA = 0.5
CONTINUETRAIN = True
NEEDTEST = False
NEEDSAVE = True
NEEDSHOW = False
EPOCH = 1
BATCHSIZE = 25
NUMWORKERS = 1
LR = 1e-3
ISCUDA = True
RECORDPOINT = 1
TESTPOINT = 100


def makedir(path):
    '''
    如果文件夹不存在，就创建
    :param path:路径
    :return:路径名
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Trainer:
    def __init__(self, net, netfile_name, cfgfile=None):
        self.net = net
        self.netfile_name = netfile_name

        makedir(SAVE_DIR)

        net_savefile = "{0}.{1}".format(self.netfile_name, NETFILE_EXTENTION)
        self.save_dir = os.path.join(SAVE_DIR, "nets")
        makedir(self.save_dir)
        self.save_path = os.path.join(self.save_dir, net_savefile)

        if os.path.exists(self.save_path) and CONTINUETRAIN:
            try:
                self.net.load_state_dict(torch.load(self.save_path))
                print("net param load successful")
            except:
                self.net = torch.load(self.save_path)
                print("net load successful")
        else:
            self.net.paraminit()
            print("param initial complete")

        self.logdir = os.path.join(SAVE_DIR, "log")
        makedir(self.logdir)
        self.logdictfile = os.path.join(self.logdir, "{0}.log".format(self.netfile_name))

        if os.path.exists(self.logdictfile) and CONTINUETRAIN:
            self.resultdict = torch.load(self.logdictfile)
            print("log load successfully")
        else:
            self.resultdict = {}

        self.size = {"Pnet": 12, "Rnet": 24, "Onet": 48}[self.net.name]
        self.cls_loss = nn.BCELoss()
        self.offset_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters())

        if ISCUDA:
            self.net = self.net.to(DEVICE)

        print("initial complete")

    def loss_fn(self, output_cls, output_offset, cls, offset):

        cls_loss = self.cls_loss(output_cls, cls)
        offset_loss = self.offset_loss(offset, output_offset)
        loss = ALPHA * cls_loss + (1 - ALPHA) * offset_loss
        return loss, cls_loss, offset_loss

    def scalarplotting(self, key: str = 'loss', j=0):
        '''

        :param datas:
        :param key: 要查询的键
        :param j: 采样点
        :return:
        '''
        datas = torch.load(self.logdictfile)

        save_dir = os.path.join(SAVE_DIR, key)
        makedir(save_dir)
        save_name = "{0}.jpg".format(key)

        save_file = os.path.join(save_dir, save_name)
        values = []
        if len(datas) > 0:
            for i, j_values in datas.items():
                values.append(j_values[j][key])
        if len(values) != 0:
            plt.plot(values)
            plt.savefig(save_file)
            plt.show()

    def FDplotting(self, bins=10):
        save_dir = os.path.join(SAVE_DIR, "params")
        makedir(save_dir)
        save_name = "{0}_param.jpg".format(self.netfile_name)
        save_file = os.path.join(SAVE_DIR, save_name)
        params = []
        for param in self.net.parameters():
            params.extend(param.view(-1).cpu().detach().numpy())
        params = np.array(params)
        histo = np.histogram(params, bins, range=(np.min(params), np.max(params)))
        plt.plot(histo[1][1:], histo[0])
        plt.savefig(save_file)
        plt.show()

    def train(self):
        start_time = time.time()
        dataset = Dataset(LABEL_DIR, PIC_DIR, self.size)
        dataloader = data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True,
                                     num_workers=NUMWORKERS,
                                     drop_last=True)
        dataloader_len = len(dataloader)
        print("dataloader_len", dataloader_len)
        i = len(self.resultdict)
        print("i1", i)
        if i == 0:
            j = 0
        else:
            j = len(self.resultdict[i - 1])
            if j >= dataloader_len:
                j = 0
            else:
                i -= 1
                print("i2", i, j, dataloader_len - 1)

        print("i3 ", i, j)

        flag = False
        while i < EPOCH:
            if flag:
                break
            self.net.train()

            if j == 0:
                self.resultdict[i] = {}
            for img_data_, cls_, offset_ in dataloader:

                self.net.train()
                if ISCUDA:
                    img_data_ = img_data_.to(DEVICE)
                    cls_ = cls_.to(DEVICE)
                    offset_ = offset_.to(DEVICE)
                _output_cls, _output_offset = self.net(img_data_)
                _output_cls = _output_cls.view(-1, 1)
                _output_offset = _output_offset.view(-1, 4)

                cls_mask = torch.lt(cls_[:, 0], 2)
                offset_mask = torch.gt(cls_[:, 0], 0)

                cls = cls_[cls_mask]
                offset = offset_[offset_mask]

                output_cls = _output_cls[cls_mask]
                output_offset = _output_offset[offset_mask]

                loss, cls_loss, offset_loss = self.loss_fn(output_cls, output_offset, cls, offset)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                params = []
                for param in self.net.parameters():
                    params.extend(param.view(-1).data)

                checktime = time.time() - start_time
                cls_acc = torch.mean(torch.lt(torch.abs(torch.sub(cls, output_cls)), 0.02).float())
                self.resultdict[i][j] = {"loss": loss.detach().cpu(), "cls_loss": cls_loss.detach().cpu(),
                                         "offset_loss": offset_loss.detach().cpu(), "cls_acc": cls_acc.detach().cpu(),
                                         "time": time.time()}

                if j % RECORDPOINT == 0:

                    offset_acc = torch.mean(torch.lt(torch.abs(torch.sub(offset, output_offset)), 0.02).float())

                    result = "{'epoch':%d,'batch':%d,'loss':%f,'cls_loss':%f,'offset_loss':%f,'total_time':%.2f,'cls_acc':%f,'offset_acc':%f,'time':%s}" % (
                        i, j, loss, cls_loss, offset_loss, checktime, cls_acc, offset_acc,
                        time.strftime("%Y%m%d%H%M%S", time.localtime()))
                    print(result)

                    if NEEDSAVE:
                        torch.save(self.net.state_dict(), self.save_path)
                        torch.save(self.resultdict, self.logdictfile)
                        print("net save successful")

                if i > 5 and j >= 190:
                    flag = True
                    break

                if j >= dataloader_len - 1:
                    j = 0
                    break
                else:
                    j += 1
            i += 1



if __name__ == '__main__':
    net = ONet()
    # net.paraminit()
    # trainer = Trainer(net, save_path_o, label_path_48, pic_path_48, isCuda=True)
    # trainer.train()
    trainer = Trainer(net, netfile_name="onet_00_0")
    trainer.train()
    # trainer.scalarplotting()
    # trainer.FDplotting()
    # run()
    # trainer.scalarplotting(trainer.getstatistics(), "loss")
    # trainer.FDplotting(net)
