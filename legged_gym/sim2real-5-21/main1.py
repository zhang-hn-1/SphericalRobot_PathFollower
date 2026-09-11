import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
import os

# 显式匹配 sim 和 real 的文件路径对
SIM_REAL_FILE_PAIRS = [
    ("sim_data/sim_01.json", "real_data/real_01.json"),
    ("sim_data/sim_02.json", "real_data/real_02.json"),
    ("sim_data/sim_03.json", "real_data/real_03.json"),
    ("sim_data/sim_04.json", "real_data/real_04.json"),
    # 可以继续添加更多对
]

# ------------------ 配置 ------------------
STATE_KEYS = [
    "euler_x", "euler_y", "euler_z",
    "lin_vel_x", "lin_vel_y", "lin_vel_z",
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "v1", "pos2"
]  # 使用的状态维度键
WINDOW_SIZE = 5  # 时间窗口长度
EPOCHS = 100  # 训练轮数

# ------------------ 数据加载 ------------------
def load_json_to_array(file_path, selected_keys):
    """从JSON文件加载状态数组"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array([[entry[k] for k in selected_keys] for entry in data])

def create_sequence_dataset(sim_states, real_states, window_size):
    """构建序列输入和残差标签数据集（每对 sim/real 单独切片）"""
    sim_seq, residual_targets = [], []
    for i in range(window_size - 1, len(sim_states)):
        window = sim_states[i - window_size + 1: i + 1]
        sim_seq.append(window)
        residual = real_states[i] - sim_states[i]  # 真实值 - 仿真值
        residual_targets.append(residual)
    return np.array(sim_seq), np.array(residual_targets)

# def load_and_process_all_sequences(sim_dir, real_dir, selected_keys, window_size):
#     """批量加载所有 sim/real 文件，并构建完整数据集"""
#     sim_files = sorted(glob.glob(os.path.join(sim_dir, '*.json')))
#     real_files = sorted(glob.glob(os.path.join(real_dir, '*.json')))
#     assert len(sim_files) == len(real_files), "sim 和 real 文件数量不一致"

#     X_all, y_all = [], []
#     for sim_f, real_f in zip(sim_files, real_files):
#         sim = load_json_to_array(sim_f, selected_keys)
#         real = load_json_to_array(real_f, selected_keys)
#         X_seq, y_seq = create_sequence_dataset(sim, real, window_size)
#         X_all.append(X_seq)
#         y_all.append(y_seq)
#     return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)
def load_and_process_all_sequences(file_pairs, selected_keys, window_size):
    """从指定的 sim/real 文件路径对加载并处理数据"""
    X_all, y_all = [], []
    for sim_path, real_path in file_pairs:
        sim = load_json_to_array(sim_path, selected_keys)
        real = load_json_to_array(real_path, selected_keys)
        X_seq, y_seq = create_sequence_dataset(sim, real, window_size)
        X_all.append(X_seq)
        y_all.append(y_seq)
    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)


# 批量加载仿真与实地状态数据（多个文件）
# X, y = load_and_process_all_sequences("sim_data", "real_data", STATE_KEYS, WINDOW_SIZE)
X, y = load_and_process_all_sequences(SIM_REAL_FILE_PAIRS, STATE_KEYS, WINDOW_SIZE)
print(f"加载数据完成：X.shape = {X.shape}, y.shape = {y.shape}")
print(f"X[0]: {X[0]}")

# ------------------ 数据预处理 ------------------
scaler_input = StandardScaler()
scaler_output = StandardScaler()
X_norm = scaler_input.fit_transform(X.reshape(-1, len(STATE_KEYS))).reshape(X.shape)
y_norm = scaler_output.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)

# ------------------ Transformer模型 ------------------
class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerResidualCorrector(nn.Module):
    """Transformer网络，用于残差预测"""
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, input_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.mlp_head(x[:, -1, :])  # 取最后一个时间步的输出作为预测

# ------------------ 模型训练 ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerResidualCorrector(input_dim=len(STATE_KEYS)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    model.train()
    pred = model(X_train_t)
    loss = loss_fn(pred, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(X_test_t), y_test_t).item()
    test_losses.append(test_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Test Loss = {test_loss:.4f}")

# ------------------ 损失曲线可视化 ------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", color='blue')
plt.plot(test_losses, label="Test Loss", color='orange')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("训练与测试损失曲线")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 保存模型
# 保存模型参数
# torch.save(model.state_dict(), "model.pth")
jit_model = torch.jit.script(model)  # Use torch.jit.trace if the model has dynamic control flow
jit_model.save("model_jit.pth")

# 保存 scaler 对象
with open("scaler_input.pkl", "wb") as f_in:
    pickle.dump(scaler_input, f_in)

with open("scaler_output.pkl", "wb") as f_out:
    pickle.dump(scaler_output, f_out)

print("模型与 scaler 已保存。")


# ------------------ 状态修正接口 ------------------
def correct_state_with_transformer(sim_state_window):
    """给定一段窗口的仿真状态，预测修正后的状态"""
    assert len(sim_state_window) == WINDOW_SIZE
    sim_array = np.array([[s[k] for k in STATE_KEYS] for s in sim_state_window])
    sim_norm = scaler_input.transform(sim_array).reshape(1, WINDOW_SIZE, -1)
    with torch.no_grad():
        sim_tensor = torch.tensor(sim_norm, dtype=torch.float32).to(device)
        pred_residual = model(sim_tensor).cpu().numpy()
    corrected = sim_array[-1] + scaler_output.inverse_transform(pred_residual)
    return corrected[0]

# ------------------ 示例 ------------------
# 示例目录结构要求：sim_data/real_data 文件夹下各有多个 .json 序列文件
# 使用示例：
# sim_state_window = [sim_data[i] for i in range(WINDOW_SIZE)]
# corrected_state = correct_state_with_transformer(sim_state_window)
# print(corrected_state)
