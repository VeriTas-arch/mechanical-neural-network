import matplotlib.pyplot as plt
import numpy as np
 
# 创建空的图形对象
fig = plt.figure()
ax = fig.add_subplot(111)
 
# 初始化数据列表
x = []
y = []

plt.ion()
plt.show()

for i in range(10):
    # 模拟生成随机数作为 x、y 值
    random_x = np.random.rand(10, 1)
    random_y = np.random.rand(10, 1)
    
    # 将新生成的点添加到数据列表中
    x.append(random_x)
    y.append(random_y)
    
    # 清除之前的所有点并重新绘制
    ax.clear()
    ax.scatter(x, y)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('EVA')
    
    # 更新图像显示
    plt.draw()
    plt.pause(0.2)

plt.ioff()

plt.show()