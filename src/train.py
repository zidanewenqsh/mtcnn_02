import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import argparse
import time
from torch.utils import data
import matplotlib.pyplot as plt
import configparser
from src.nets import PNet, RNet, ONet, Net
from dataset.datasets import Dataset
from detect.detectors import Detector
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)



# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CFGFILE = r".\cfg.ini"

SAVE_DIR = r""
PIC_DIR = r""
LABEL_DIR = r""

ALPHA = 0.5

CONTINUETRAIN = False
NEEDTEST = False
NEEDSAVE = False
NEEDSHOW = False
EPOCH = 1
BATCHSIZE = 1
NUMWORKERS = 1
LR = 1e-3
ISCUDA = True
SAVEDIR_EPOCH = r""
TEST_IMG = r""
PNET_TRAINED = r""
RNET_TRAINED = r""
ONET_TRAINED = r""
RECORDPOINT = 10
TESTPOINT = 100


class Trainer:
    def __init__(self, netfile_name: str, cfgfile=r".\cfg.ini"):
        # self.net = net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parser = argparse.ArgumentParser(description="base class for network training")
        self.args = self.__argparser(parser)

        if self.args.name == None:
            self.netfile_name = netfile_name
        else:
            self.netfile_name = self.args.name

        self.__cfginit(cfgfile)

        self.__makedir(SAVE_DIR)

        id = self.netfile_name.split("_")[0]
        # self.net = {"pnet":PNet(), "rnet":RNet(), "onet":ONet()}[self.netfile_name.split("_")[0]]
        print("id: ", id)
        if id == 'pnet':
            self.net = PNet()
        elif id == 'rnet':
            self.net = RNet()
        elif id == 'onet':
            self.net = ONet()
        else:
            raise ValueError

        net_savefile = "{0}.pth".format(self.netfile_name)
        netparam_savefile = "{0}.pt".format(self.netfile_name)
        self.save_dir = os.path.join(SAVE_DIR, "nets")
        self.__makedir(self.save_dir)
        self.save_path = os.path.join(self.save_dir, net_savefile)
        self.save_path_p = os.path.join(self.save_dir, netparam_savefile)
        self.__makedir(SAVEDIR_EPOCH)
        self.savepath_epoch = os.path.join(SAVEDIR_EPOCH, net_savefile)

        if CONTINUETRAIN:
            if os.path.exists(self.save_path):
                self.net = torch.load(self.save_path)
                print("net load successful")
            elif os.path.exists(self.save_path_p):
                self.net.load_state_dict(torch.load(self.save_path_p, map_location=self.device))
                print("net param load successful")
        else:
            self.net.paraminit()
            print("param initial complete")

        self.logdir = os.path.join(SAVE_DIR, "log")
        self.__makedir(self.logdir)
        self.logdictfile = os.path.join(self.logdir, "{0}.log".format(self.netfile_name))

        if os.path.exists(self.logdictfile) and CONTINUETRAIN:
            self.resultdict = torch.load(self.logdictfile)
            print("log load successfully")
        else:
            self.resultdict = {}

        # self.size = {"pnet": 12, "rnet": 24, "onet": 48}[self.net.name]
        self.cls_loss = nn.BCELoss()
        self.offset_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters())

        if ISCUDA:
            self.net = self.net.to(self.device)

        if NEEDTEST:
            self.detecter = Detector(
                returnnet=self.net.name,
                trainnet=self.net
            )

        print("initial complete")

    def __makedir(self, path):
        '''
        如果文件夹不存在，就创建
        :param path:路径
        :return:路径名
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def __cfginit(self, cfgfile):
        config = configparser.ConfigParser()
        config.read(cfgfile)
        print(self.netfile_name)
        items_ = config.items(self.netfile_name)
        for key, value in items_:
            if key.upper() in globals().keys():
                try:
                    globals()[key.upper()] = config.getint(self.netfile_name, key.upper())
                except:
                    try:
                        globals()[key.upper()] = config.getfloat(self.netfile_name, key.upper())
                    except:
                        try:
                            globals()[key.upper()] = config.getboolean(self.netfile_name, key.upper())
                        except:
                            globals()[key.upper()] = config.get(self.netfile_name, key.upper())

    def __argparser(self, parser):
        parser.add_argument("-f", "--name", type=str, default=None, help="the netfile name to train")
        return parser.parse_args()

    def __loss_fn(self, output_cls, output_offset, cls, offset):

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
        self.__makedir(save_dir)
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
        self.__makedir(save_dir)
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
        dataset = Dataset(LABEL_DIR, PIC_DIR, self.net.size)
        dataloader = data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True,
                                     num_workers=NUMWORKERS, drop_last=True)
        dataloader_len = len(dataloader)

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
        # for i in range(self.args.epoch):
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
                    img_data_ = img_data_.to(self.device)
                    cls_ = cls_.to(self.device)
                    offset_ = offset_.to(self.device)
                _output_cls, _output_offset = self.net(img_data_)
                _output_cls = _output_cls.view(-1, 1)
                _output_offset = _output_offset.view(-1, 4)

                cls_mask = torch.lt(cls_[:, 0], 2)
                offset_mask = torch.gt(cls_[:, 0], 0)

                cls = cls_[cls_mask]
                offset = offset_[offset_mask]

                output_cls = _output_cls[cls_mask]
                output_offset = _output_offset[offset_mask]

                loss, cls_loss, offset_loss = self.__loss_fn(output_cls, output_offset, cls, offset)

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
                    # print(self.net.mark)

                    if NEEDSAVE:
                        torch.save(self.net, self.save_path)
                        print("net save successful")
                        torch.save(self.resultdict, self.logdictfile)
                if j % (RECORDPOINT + 1) == 0:
                    if NEEDSAVE:
                        torch.save(self.net.state_dict(), self.save_path_p)
                        print("netparam save successful")
                        torch.save(self.resultdict, self.logdictfile)

                if NEEDTEST and j % TESTPOINT == 0 and j != 0:
                    self.test(i, j)

                if i > 5 and j >= 190:
                    flag = True
                    break

                if j >= dataloader_len - 1:
                    j = 0
                    break
                else:
                    j += 1
            if NEEDSAVE:
                torch.save(self.net.state_dict(), self.savepath_epoch)
                print("an epoch save successful")
            i += 1

    def test(self, i, j):
        with torch.no_grad():
            self.net.eval()
            # img = Image.open(TEST_IMG)
            img_name = f"{i}_{j}.jpg"
            testpic_savedir = os.path.join(SAVE_DIR, "testpic", self.netfile_name)
            self.__makedir(testpic_savedir)
            self.detecter.image_genarate(TEST_IMG, savedir=testpic_savedir if NEEDSAVE else None, img_name=img_name,
                                         show=NEEDSHOW)


if __name__ == '__main__':
    print("hello")
    trainer = Trainer(netfile_name="onet_00_0")
    # trainer.train()
