import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 主轴电流和力矩数据
current = np.array([100, 150, 200, 250, 300])
torque = np.array([3.5, 6.5, 13, 20, 26])

current2 = np.array([ 100, 150, 200  ])
torque2 = np.array([ 3.75, 8.75, 13.5  ])

current3 = np.array([-180, -120 ])
torque3 = np.array([-7.25, -3.5 ])

# 定义二次函数
def quadratic_function3(x, a, b, c, d):
    return a*x**3 + b * x**2 + c * x + d

# 定义一次函数
def quadratic_function1(x, a, b):
    return  a * x + b

def quadratic_function2(x, a, b, c):
    return a*x**2 + b * x + c 

# 拟合二次函数
params, params_covariance = curve_fit(quadratic_function1, torque3, current3)

# 打印拟合参数
print(f"拟合参数: a = {params[0]}, b = {params[1]}")

# 绘制数据点和拟合曲线
plt.figure(figsize=(8, 6))
plt.scatter(torque3, current3, label='Data Points')
plt.plot(torque3, quadratic_function1(torque3, *params), label='Fitted Curve', color='red')
plt.xlabel('Torque (N·m)')
plt.ylabel('Current (A)')
plt.title('Quadratic Fit of Current vs Torque')
plt.legend()
plt.grid(True)
plt.show()