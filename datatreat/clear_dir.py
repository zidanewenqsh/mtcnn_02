import shutil
import time
path_todel = r"F:\Dataset\mctnn_dataset\save_10261_20200114\pic"
try:
    t1 = time.time()
    shutil.rmtree(path_todel)
    t2 = time.time()
    print(t2-t1)
except Exception as e:
    print("error: ",e)