import json

def txt_to_json(input_file, output_file):
    fields = [
        "time",
        "roll",
        "pitch",
        "yaw",
        "vx",
        "vy",
        "vz",
        "wx",
        "wy",
        "wz",
        "first_vel",
        "first_cur",
        "second_vel",
        "second_pos",
        "second_cur",
        "first",
        "second"
    ]

    data = []
    initial_time = None  # 存储初始时间戳
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            # 验证字段数量
            if len(parts) != len(fields):
                print(f"警告：第 {line_num} 行包含 {len(parts)} 个字段（应为 {len(fields)}），已跳过")
                continue
            
            # 创建数据条目
            try:
                entry = {field: float(parts[i]) for i, field in enumerate(fields)}
                
                # 时间归零处理
                if initial_time is None:  # 第一个有效数据行
                    initial_time = entry["time"]
                    entry["time"] = 0.0  # 第一个时间设为0
                else:
                    entry["time"] = round(entry["time"] - initial_time, 6)  # 保留6位小数
                
                data.append(entry)
            except ValueError as e:
                print(f"第 {line_num} 行数据转换错误：{e}，已跳过")

    # 写入JSON文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"转换完成！共转换 {len(data)} 条数据，时间基准：{initial_time}")

if __name__ == "__main__":
    txt_to_json("D:/study/code/sim2real-5-17/data2convert/state_data_raw_3.txt", "output3_data.json")