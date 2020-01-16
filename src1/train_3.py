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
from nets import PNet, RNet, ONet, Net
from datasets import Dataset
from detectors import Detector

# ModuleNotFoundError: No module named 'src'
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class Trainer:
    def __init__(self, netfile_name: str, cfgfile=r".\cfg.ini"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parser = argparse.ArgumentParser(description="base class for network training")
        self.args = self.__argparser(parser)

        if self.args.name == None:
            self.netfile_name = netfile_name
        else:
            self.netfile_name = self.args.name

        self.__cfginit(cfgfile)

        id = self.netfile_name.split("_")[0]
        print("id: ", id)
        if id == 'pnet':
            self.net = PNet()
        elif id == 'rnet':
            self.net = RNet()
        elif id == 'onet':
            self.net = ONet()
        else:
            raise ValueError

        self.__makedir(self.SAVE_DIR)
        net_savefile = "{0}.pth".format(self.netfile_name)
        netparam_savefile = "{0}.pt".format(self.netfile_name)
        self.save_dir = os.path.join(self.SAVE_DIR, "nets")
        self.__makedir(self.save_dir)
        self.save_path = os.path.join(self.save_dir, net_savefile)
        self.save_path_p = os.path.join(self.save_dir, netparam_savefile)
        self.__makedir(self.SAVEDIR_EPOCH)
        self.savepath_epoch = os.path.join(self.SAVEDIR_EPOCH, net_savefile)

        if self.CONTINUETRAIN:
            if os.path.exists(self.save_path):
                self.net = torch.load(self.save_path)
                print("net load successful")
            elif os.path.exists(self.save_path_p):
                self.net.load_state_dict(torch.load(self.save_path_p, map_location=self.device))
                print("net param load successful")
        else:
            self.net.paraminit()
            print("param initial complete")

        self.logdir = os.path.join(self.SAVE_DIR, "log")
        self.__makedir(self.logdir)
        self.logdictfile = os.path.join(self.logdir, "{0}.log".format(self.netfile_name))

        if os.path.exists(self.logdictfile) and self.CONTINUETRAIN:
            self.resultdict = torch.load(self.logdictfile)
            print("log load successfully")
        else:
            self.resultdict = {}

        self.cls_loss = nn.BCELoss()
        self.offset_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters())

        if self.ISCUDA:
            self.net = self.net.to(self.device)

        if self.NEEDTEST:
            self.detecter = Detector(
                returnnet=self.net.name,
                pnet_f=self.PNET_TRAINED,
                rnet_f=self.RNET_TRAINED,
                onet_f=self.ONET_TRAINED,
                trainnet=self.net,
                isCuda=self.ISCUDA
            )

        print("initial complete")

        if self.FUNC == 2 and os.path.exists(self.logdictfile):
            scalarkey = self.args.scalarkey if self.args.scalarkey else "loss"
            self.scalarplotting(scalarkey)
            self.FDplotting()
            print("ok")

        if self.FUNC == 1:
            self.train()

    def __cfginit(self, cfgfile):
        config = configparser.ConfigParser()
        config.read(cfgfile)
        self.SAVE_DIR = config.get(self.netfile_name, "SAVE_DIR")
        self.PIC_DIR = config.get(self.netfile_name, "PIC_DIR")
        self.LABEL_DIR = config.get(self.netfile_name, "LABEL_DIR")
        self.SAVEDIR_EPOCH = config.get(self.netfile_name, "SAVEDIR_EPOCH")
        self.ISCUDA = config.getboolean(self.netfile_name, "ISCUDA")
        self.NEEDTEST = config.getboolean(self.netfile_name, "NEEDTEST")
        self.CONTINUETRAIN = config.getboolean(self.netfile_name, "CONTINUETRAIN")
        self.NEEDSAVE = config.getboolean(self.netfile_name, "NEEDSAVE")

        self.EPOCH = self.args.epoch if self.args.epoch else config.getint(self.netfile_name, "EPOCH")
        # self.ALPHA = config.getfloat(self.netfile_name, "ALPHA")
        self.ALPHA = self.args.alpha if self.args.alpha else config.getfloat(self.netfile_name, "ALPHA")
        self.FUNC = self.args.func if self.args.func else config.getint(self.netfile_name, "FUNC")

        self.PNET_TRAINED = config.get(self.netfile_name, "PNET_TRAINED")
        self.RNET_TRAINED = config.get(self.netfile_name, "RNET_TRAINED")
        self.ONET_TRAINED = config.get(self.netfile_name, "ONET_TRAINED")
        self.NEEDSHOW = config.getboolean(self.netfile_name, "NEEDSHOW")
        self.TEST_IMG = config.get(self.netfile_name, "TEST_IMG")

        self.BATCHSIZE = self.args.batchsize if self.args.batchsize else config.getint(self.netfile_name,
                                                                                            "BATCHSIZE")
        self.NUMWORKERS = self.args.numworkers if self.args.numworkers else config.getint(self.netfile_name,
                                                                                               "NUMWORKERS")
        self.RECORDPOINT = self.args.recordpoint if self.args.recordpoint else config.getint(self.netfile_name,
                                                                                                  "RECORDPOINT")
        self.TESTPOINT = self.args.testpoint if self.args.testpoint else config.getint(self.netfile_name,
                                                                                            "TESTPOINT")

    def __makedir(self, path):
        '''
        如果文件夹不存在，就创建
        :param path:路径
        :return:路径名
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def __argparser(self, parser):
        parser.add_argument("-n", "--name", type=str, default=None, help="the netfile name to train")
        parser.add_argument("-e", "--epoch", type=int, default=1, help="number of epochs")
        parser.add_argument("-b", "--batchsize", type=int, default=None, help="mini-batch size")
        parser.add_argument("-w", "--numworkers", type=int, default=None,
                            help="number of threads used during batch generation")
        parser.add_argument("-l", "--lr", type=float, default=None, help="learning rate for gradient descent")
        parser.add_argument("-r", "--recordpoint", type=int, default=None, help="print frequency")
        parser.add_argument("-t", "--testpoint", type=int, default=None,
                            help="interval between evaluations on validation set")
        parser.add_argument("-a", "--alpha", type=float, default=None, help="ratio of conf and offset loss")
        parser.add_argument("-f", "--func", type=int, default=None, help="the choice of function")
        parser.add_argument("-k", "--scalarkey", type=str, default=None, help="the choice of scalar to plot")
        return parser.parse_args()

    def __loss_fn(self, output_cls, output_offset, cls, offset):

        cls_loss = self.cls_loss(output_cls, cls)
        offset_loss = self.offset_loss(offset, output_offset)
        loss = self.ALPHA * cls_loss + (1 - self.ALPHA) * offset_loss
        return loss, cls_loss, offset_loss

    def scalarplotting(self, key: str = 'loss', j=0):
        '''

        :param datas:
        :param key: 要查询的键
        :param j: 采样点
        :return:
        '''
        datas = torch.load(self.logdictfile)

        save_dir = os.path.join(self.SAVE_DIR, "statistics")
        self.__makedir(save_dir)
        save_name = "{0}.jpg".format(key)

        save_file = os.path.join(save_dir, save_name)
        values = []
        if len(datas) > 0:
            for i, j_values in datas.items():
                values.append(j_values[j][key])

        if len(values) != 0:
            plt.clf()
            plt.plot(values)
            plt.savefig(save_file)

        if self.NEEDSHOW:
            plt.show()
            plt.pause(0.1)

    def FDplotting(self, bins=10):
        save_dir = os.path.join(self.SAVE_DIR, "statistics")
        self.__makedir(save_dir)
        save_name = "{0}_param.jpg".format(self.netfile_name)
        save_file = os.path.join(save_dir, save_name)
        params = []
        for param in self.net.parameters():
            params.extend(param.view(-1).cpu().detach().numpy())
        params = np.array(params)
        histo = np.histogram(params, bins, range=(np.min(params), np.max(params)))

        plt.clf()
        plt.plot(histo[1][1:], histo[0])
        plt.savefig(save_file)

        if self.NEEDSHOW:
            plt.show()
            plt.pause(0.1)

    def train(self):
        start_time = time.time()
        dataset = Dataset(self.LABEL_DIR, self.PIC_DIR, self.net.size)
        dataloader = data.DataLoader(dataset, batch_size=self.BATCHSIZE, shuffle=True,
                                     num_workers=self.NUMWORKERS, drop_last=True)
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
        while i < self.EPOCH:
            if flag:
                break
            self.net.train()

            if j == 0:
                self.resultdict[i] = {}

            for img_data_, cls_, offset_ in dataloader:

                self.net.train()
                if self.ISCUDA:
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

                if j % self.RECORDPOINT == 0:

                    offset_acc = torch.mean(torch.lt(torch.abs(torch.sub(offset, output_offset)), 0.02).float())

                    result = "{'epoch':%d,'batch':%d,'loss':%f,'cls_loss':%f,'offset_loss':%f,'total_time':%.2f,'cls_acc':%f,'offset_acc':%f,'time':%s}" % (
                        i, j, loss, cls_loss, offset_loss, checktime, cls_acc, offset_acc,
                        time.strftime("%Y%m%d%H%M%S", time.localtime()))
                    print(result)
                    # print(self.net.mark)

                    if self.NEEDSAVE:
                        torch.save(self.net, self.save_path)
                        print("net save successful")
                        torch.save(self.resultdict, self.logdictfile)
                if j % (self.RECORDPOINT + 1) == 0:
                    if self.NEEDSAVE:
                        torch.save(self.net.state_dict(), self.save_path_p)
                        print("netparam save successful")
                        torch.save(self.resultdict, self.logdictfile)

                if self.NEEDTEST and j % self.TESTPOINT == 0 and j != 0:
                    self.test(i, j)

                if i > 5 and j >= 190:
                    flag = True
                    break

                if j >= dataloader_len - 1:
                    j = 0
                    break
                else:
                    j += 1
            if self.NEEDSAVE:
                torch.save(self.net.state_dict(), self.savepath_epoch)
                print("an epoch save successful")
            i += 1

    def test(self, i, j):
        with torch.no_grad():
            self.net.eval()
            # img = Image.open(TEST_IMG)
            img_name = f"{i}_{j}.jpg"
            testpic_savedir = os.path.join(self.SAVE_DIR, "testpic", self.netfile_name)
            self.__makedir(testpic_savedir)
            self.detecter.image_genarate(self.TEST_IMG, savedir=testpic_savedir if self.NEEDSAVE else None,
                                         img_name=img_name, show=self.NEEDSHOW)


if __name__ == '__main__':
    trainer = Trainer(netfile_name="onet_00_0")
    # trainer.train()
    trainer.FDplotting()
    trainer.scalarplotting()
    trainer.FDplotting()
