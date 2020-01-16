import torch

path = r"F:\Dataset\mctnn_dataset\save_10261_20200114\label\label_12.torch"
path1 = r"F:\Dataset\mctnn_dataset\save_10261_20200114\data\data_12.torch"
a = torch.load(path)
b = torch.load(path1)
# print(a)
print(len(a))
posi = 0
part = 0
nega = 0

for k, v in a.items():

    if v[0] == 0:
        nega += 1
    elif v[0] == 1:
        posi += 1
    elif v[0] == 2:
        part += 1
    print(v)
print(posi, part, nega)

for k, v in b.items():
    print(k)
    print(v.shape)
    # print(v)
    break
