import numpy as np

def read_and_interpolate_state_data1(file_path):
    """
    读取 state_data_raw_1.txt 文件并解析数据，同时对缺失的时间点进行插值。
    返回:
        list: 包含每行数据的列表，每行数据是一个字典，值为 float 格式。
    """
    data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        # 跳过表头（第一行）
        header = lines[0].strip().split()
        for line in lines[1:]:
            # 跳过空行或注释行
            if line.strip() == "" or line.startswith("//"):
                continue
            # 解析数据行并将值转换为 float
            values = [float(value) for value in line.strip().split()]
            data.append(dict(zip(header, values)))

    if data:
        # 将时间从 0 开始，并保留两位小数
        initial_time = data[0]["time"]
        for row in data:
            row["time"] = round(row["time"] - initial_time, 2)
        # 提取时间序列
        times = [row["time"] for row in data]
        final_time = times[-1]

        # 生成完整的时间序列（间隔为 0.02）
        full_times = np.arange(0, final_time + 0.02, 0.02)

        # 创建插值后的数据列表
        interpolated_data = []
        for t in full_times:
            if t in times:
                # 如果时间点存在，直接使用原始数据
                row = next(row for row in data if row["time"] == t)
                interpolated_data.append(row)
            else:
                # 如果时间点缺失，进行插值
                prev_row = next(row for row in reversed(data) if row["time"] < t)
                next_row = next(row for row in data if row["time"] > t)
                if next_row is None:
                    print(f"No row found with time > {t}")
                    continue  # 或者根据逻辑处理

                # 插值每个字段
                interpolated_row = {"time": round(t, 2)}
                for key in prev_row.keys():
                    if key != "time":
                        prev_value = prev_row[key]
                        next_value = next_row[key]
                        interpolated_row[key] = prev_value + (next_value - prev_value) * ((t - prev_row["time"]) / (next_row["time"] - prev_row["time"]))
                interpolated_data.append(interpolated_row)

        return interpolated_data
    
def read_and_process_state_data2(file_path, total_time=12.0, time_interval=0.02):
    """
    读取 state_data_raw_i.txt 文件，并处理时间项从 0 开始，保留两位小数。
    每隔 0.02s 取一行数据。
    参数:
        file_path (str): 文件路径。
    返回:
        list: 处理后的数据列表。
    """
    data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        # 跳过表头（第一行）
        header = lines[0].strip().split()
        for line in lines[1:]:
            # 跳过空行或注释行
            if line.strip() == "" or line.startswith("//"):
                continue
            # 解析数据行并将值转换为 float
            values = [float(value) for value in line.strip().split()]
            data.append(dict(zip(header, values)))
    # 确保时间项从 0 开始
    min_time = data[0]["time"]
    for row in data:
        row["time"] = round(row["time"] - min_time, 2)

    # 以 0.02s 的时间间隔取数据
    processed_data = []
    full_times = np.arange(0, total_time, time_interval)
    for t in full_times:
        # 找到最接近的时间点
        closest_row = min(data, key=lambda row: abs(row["time"] - t))
        processed_data.append(closest_row)

    return processed_data


# 文件路径
file_path = "data/state_data_raw_i.txt"

# 读取并插值数据
state_data = read_and_process_state_data2(file_path)

# 示例：打印前5行数据
print("前5行数据:")
for i, row in enumerate(state_data[-5:]):
    print(f"第 {i+1} 行数据: {row}")

print(f"state_data 的长度: {len(state_data)}")