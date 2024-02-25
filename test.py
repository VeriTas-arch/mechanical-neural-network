import numpy as np

def ret(x, y):
    return x, y

# 创建数组（3维）
a = np.arange(100).reshape((10, 5, 2))

arr = np.array([1, 2, 3, 4, 5])
sort_fitness = np.argsort(arr)
point = np.random.choice(5, p=sort_fitness / sum(sort_fitness))
print(point)
