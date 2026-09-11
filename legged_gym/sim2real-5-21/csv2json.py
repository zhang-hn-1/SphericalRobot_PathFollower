import csv
import json

# 文件路径
csv_file_path = r'D:/study/code/sim2real-5-17/data2convert/state_data_sim_4.csv'
json_file_path = 'sim_data/sim_04.json'

# 不需要保留的字段
excluded_fields = {'v2', 'pos1', 'pos_x', 'pos_y', 'quat_x', 'quat_y', 'quat_z', 'quat_w', 'torque1', 'torque2'}

# 读取 CSV 文件
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    original_headers = next(csv_reader)

    # 将第一列变为 'time'，其余字段向后移动
    new_headers = ['time'] + original_headers

    data = []
    for row in csv_reader:
        new_row = [row[0]] + row  # 插入 time 为第一列，原始 row 不变

        # 构建字典，排除不需要的字段
        entry = {}
        for key, value in zip(new_headers, new_row):
            if key in excluded_fields:
                continue  # 跳过被排除的字段
            try:
                entry[key] = float(value)
            except ValueError:
                entry[key] = value
        data.append(entry)

# 写入 JSON 文件
with open(json_file_path, mode='w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=2)

print(f"转换完成，已保存为 {json_file_path}")

