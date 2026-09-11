import json

# 输入输出文件路径
input_json_path = "output4_data.json"
output_json_path = "real_data/real_04.json"

with open(input_json_path, "r", encoding="utf-8") as f:
    input_json_list = json.load(f)  # 注意这是一个列表

# 定义字段映射关系
field_mapping = {
    "time": "time",
    "first_vel": "v1",
    "second_pos": "pos2",
    "vx": "lin_vel_x",
    "vy": "lin_vel_y",
    "vz": "lin_vel_z",
    "wx": "ang_vel_x",
    "wy": "ang_vel_y",
    "wz": "ang_vel_z",
    "roll": "euler_x",
    "pitch": "euler_y",
    "yaw": "euler_z"
}

# 转换每一个 JSON 对象
output_json_list = []
for item in input_json_list:
    new_item = {new_key: item[old_key] for old_key, new_key in field_mapping.items() if old_key in item}
    output_json_list.append(new_item)

# 写入输出文件
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(output_json_list, f, indent=2)

print(f"转换完成，已保存为 {output_json_path}")
