import numpy as np

lis = ["A","B"]
outcome = ["中文","English"]

def roll(lis):
    index_lis = np.arange(len(lis))
    new_lis = []
    np.random.shuffle(index_lis)
    new_lis.extend(list(lis[i] for i in list(index_lis)))
    return new_lis


print(dict(zip(roll(lis),outcome)))
