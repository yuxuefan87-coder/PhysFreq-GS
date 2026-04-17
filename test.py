import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 生成模拟的“劣质”观测数据 (比如带反光噪点的心跳轨迹)
# ==========================================
N_frames = 100
t = torch.linspace(0, 1, N_frames) # 时间轴 0 到 1

# 真实的物理运动：一个低频的平滑正弦波
true_trajectory = torch.sin(2 * np.pi * t) 
# 观测到的劣质数据：加入了剧烈的高频噪点 (模拟 4DGS 中 MLP 常犯的错误)
noisy_observation = true_trajectory + 0.5 * torch.randn(N_frames)

# ==========================================
# 2. 定义你的核心创新：DCT 解析物理形变模型
# ==========================================
class PIDCT_Trajectory(nn.Module):
    def __init__(self, K_coeffs=10):
        super().__init__()
        self.K = K_coeffs
        # 你的创新点：放弃 MLP，只优化 K 个频率系数 (大幅降低参数量)
        self.C = nn.Parameter(torch.zeros(self.K)) 
        
        # 预计算频率 omega: [0, pi, 2pi, 3pi...]
        self.omega = torch.arange(self.K).float() * np.pi
        
        # 预计算用于“二阶导数(加速度)”的常数惩罚项: omega^2
        # 这是你论文里“零显存开销算物理 Loss”的核心绝杀！
        self.accel_weights = self.omega ** 2 

    def forward(self, t):
        # 构造 DCT 基函数矩阵: [N_frames, K]
        # 公式: cos(w_k * t)
        basis = torch.cos(t.unsqueeze(1) * self.omega.unsqueeze(0))
        
        # 计算位置 mu(t) = sum(C_k * cos(w_k * t))
        position = torch.matmul(basis, self.C)
        
        # 解析计算加速度 a(t) = sum(C_k * (-omega^2) * cos(w_k * t))
        # 注意：这里没有任何 Autograd 计算图，就是一次常数乘法！
        acceleration = torch.matmul(basis, self.C * (-self.accel_weights))
        
        return position, acceleration

# ==========================================
# 3. 训练与优化过程
# ==========================================
model = PIDCT_Trajectory(K_coeffs=15) # 保留 15 个系数
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

lambda_pde = 0.0001# 物理加速度惩罚权重 (你可以尝试改为 0 看看灾难后果)

print("开始训练...")
for step in range(50000):
    optimizer.zero_grad()
    
    pred_pos, pred_accel = model(t)
    
    # 损失函数 1: 拟合观测数据 (重建误差)
    loss_data = torch.mean((pred_pos - noisy_observation)**2)
    # 损失函数 2: 牛顿力学惯性先验 (让加速度尽可能小，平滑轨迹)
    loss_pde = torch.mean(pred_accel**2) 
    
    # 总 Loss 缝合
    loss = loss_data + lambda_pde * loss_pde
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step} | Total Loss: {loss.item():.4f} | PDE Loss: {loss_pde.item():.4f}")

# ==========================================
# 4. 可视化你的“降维打击”效果
# ==========================================
pred_pos, _ = model(t)
plt.figure(figsize=(10, 5))
plt.plot(t.numpy(), true_trajectory.numpy(), 'g-', linewidth=3, label='True Physics Trajectory (Ground Truth)')
plt.plot(t.numpy(), noisy_observation.numpy(), 'r.', alpha=0.5, label='Noisy Observation (Failed MLP)')
plt.plot(t.numpy(), pred_pos.detach().numpy(), 'b-', linewidth=2, label='Your PI-DCT Recovery')
plt.legend()
plt.title("PI-DCT vs High-Frequency Noise (Toy Verification)")
plt.savefig("toy_result1.png", dpi=300, bbox_inches='tight')
print("图片已保存为 toy_result.png")