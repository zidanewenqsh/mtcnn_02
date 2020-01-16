import time

class Timekeeper:
    '''
    用来计算程序运行的时间
    '''
    _time = time.time()
    _interval = 0
    # _interval = 0
    def __new__(cls, *args, **kwargs):
        return super(Timekeeper,cls).__new__(cls)
    def __init__(self):
        pass
    @classmethod
    def gettime(cls):
        cls._interval = time.time()-cls._time
        return cls._interval
    @classmethod
    def reinittime(cls):
        cls._time = time.time()

    @classmethod
    def formattime(cls):
        return "%.2fh" % (cls._interval / 3600) if cls._interval >= 3600 else "%.2fmin" % (
                cls._interval / 60) if cls._interval >= 60 else "%.5fs" % cls._interval
def train(x):
    for i in range(x):
        i = i+1
        print(Timekeeper.formattime())
    return i

if __name__ == '__main__':
    # for i in range(10000000):
    #     i += 1
    #     if i%1000 ==0:
    #         print(Timekeeper.gettime())
    # print(Timekeeper.gettime())
    # print(Timekeeper.formattime())
    x = 1000
    train(x)
    print(type(time.time()))
