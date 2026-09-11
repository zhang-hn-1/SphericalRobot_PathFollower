import matplotlib.pyplot as plt
import math

Trajectory = []
episode_length = 2500
for i in range(2500):
    if i <=100:
        Trajectory.append([0 , 0.25*(i/50)*(i/50)])
    else:
        Trajectory.append([4 - 4 * math.cos((i-100)/200) , 1 + 4 * math.sin((i-100)/200)])

Circle_Trajectory = []
for i in range(350):
    if i <=20:
        Circle_Trajectory.append([0.25*(i/10)*(i/10),0])
    else:
        Circle_Trajectory.append([1 + 5 * math.sin((i-20)/50) , 5 - 5 * math.cos((i-20)/50)])
x_coords = [x[0] for x in Circle_Trajectory]
y_coords = [x[1] for x in Circle_Trajectory]

def plot_trajectory(x, y):
    """
    绘制轨迹图。

    参数:
    x (list): x坐标的列表
    y (list): y坐标的列表
    """
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title('Trajectory Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

# # 示例数据
# x_coords = [0, 1, 2, 3, 4, 5]
# y_coords = [0, 1, 4, 9, 16, 25]

# 绘制轨迹
plot_trajectory(x_coords, y_coords)
