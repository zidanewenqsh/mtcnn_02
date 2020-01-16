import time
import random


class Timekeeper:
    _time = time.time()
    _interval = 0
    # a = time.time()
    def __init__(self):
        # self.time = time
        Timekeeper._time = time.time()
    def __new__(cls):
    #     # Timekeeper.timekeeper(cls)
    #     return super(Timekeeper,cls).__new__(cls)
        return object.__new__(cls)

    @classmethod#这个很重要，要不改不了类变量
    def timekeeperinit(cls):
        '''
        类变量的重新初始化
        :return:
        '''
        cls._time = time.time()
        # cls.__time = abs(cls.__time-self.time)


    @classmethod
    def gettime(cls):
        cls._interval = time.time()-cls._time
        return cls._interval

    @classmethod
    def formattime(cls):
        return "%.2fh" % (cls._interval / 3600) if cls._interval >= 3600 else "%.2fmin" % (
                cls._interval / 60) if cls._interval >= 60 else "%.2fs" % cls._interval


if __name__ == '__main__':
    a = time.strftime("%Y%m%d,%H:%M:%S", time.localtime())
    a = time.strftime("%S", time.localtime())
    print(a)
    print(time.localtime())
    # a = Timekeeper()
    # print(a._time)
    # b = Timekeeper(time.time())
    # print(b._time)
    # c = Timekeeper(time.time())
    # print(c._time)
    # print(a.gettime())
    print(time.time())
    x = 0
    for i in range(10000000):
        x = x+1
        if i%10000000 ==0:
            print(x)
    # print(a.gettime())
    # print(Timekeeper.gettime())
    # a.timekeeperinit()
    # Timekeeper.timekeeperinit()
    print(Timekeeper.gettime())
    print(Timekeeper.formattime())
    for i in range(1000000):
        x = x+1
        if i%100000 ==0:
            print(x)
    # print(a.gettime())
    print(Timekeeper.gettime(),Timekeeper._interval,Timekeeper._time)#类在程序刚运行时就被初始化，而不是在调用的时候
    print(Timekeeper.formattime())



