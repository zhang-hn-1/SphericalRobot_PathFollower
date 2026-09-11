import numpy as np
import matplotlib.pyplot as plt

def plot_action_sequence():
    """
    绘制动作值随时间变化的曲线。
    """
    # 时间步长和总时间
    dt = 0.02  # 每个时间步的间隔
    total_time = 20.0  # 总时间 20 秒
    episode_length_bufs = np.arange(0, total_time / dt)  # 时间计数

    # 生成动作序列
    actions = []
    for episode_length_buf in episode_length_bufs:
        time = episode_length_buf * dt
        if time < 1.0:
            action1 = 0.0
        elif 1.0 <= time < 8.0:
            # 第 2 秒到第 3 秒，线性上升
            action1 = -3.0
        else:
            action1 = 0.0

        # 动作序列逻辑
        if time < 2.0:
            action2 = 0.0
        elif 2.0 <= time < 3.0:
            action2 = 0.0 + (0.4 / 2.0) * (time - 3.0)
        elif 3.0 <= time < 5.0:
            action2 = 0.4
        elif 5.0 <= time < 7.0:
            action2 = -0.3
        elif 7.0 <= time < 9.0:
            action2 = -0.3 + (0.75 / 2.0) * (time - 7.0)  # 线性上升
        elif 9.0 <= time < 10.0:
            action2 = 0.45 - (0.45 / 1.0) * (time - 9.0)  # 余弦信号
        else:
            action2 = 0.0
        actions.append(action2)

    # 绘制动作曲线
    times = episode_length_bufs * dt  # 将时间计数转换为秒
    plt.figure(figsize=(10, 6))
    plt.plot(times, actions, label="Action Value", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Action Value")
    plt.title("Action Sequence Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 调用函数绘制动作曲线
plot_action_sequence()