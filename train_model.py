import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy
import time
import os

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 自定义数据集类
class SeismicDataset(Dataset):
    def __init__(self, seismic, vp, density):
        self.seismic = seismic
        self.vp = vp
        self.density = density

    def __len__(self):
        return self.seismic.shape[0]

    def __getitem__(self, idx):
        seismic_trace = self.seismic[idx].reshape(-1, 1).astype(np.float32)
        vp_trace = self.vp[idx].reshape(-1).astype(np.float32)
        density_trace = self.density[idx].reshape(-1).astype(np.float32)
        return torch.tensor(seismic_trace), torch.tensor(vp_trace), torch.tensor(density_trace)


# 修正后的多任务UNet模型架构
class MultiTaskUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(MultiTaskUNet, self).__init__()

        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 中间层
        self.bottleneck = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, kernel_size=5, padding=2),
            nn.ReLU()
        )

        # 任务特定解码器 - Vp (修正上采样率)
        self.decoder_vp = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2, padding=1),  # 上采样2倍
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # 上采样2倍
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # 上采样2倍
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # 上采样2倍
            nn.ReLU(),
            nn.Conv1d(16, output_channels, kernel_size=1)
        )

        # 任务特定解码器 - Density (同上修正)
        self.decoder_density = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, output_channels, kernel_size=1)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 共享特征提取
        x = self.encoder(x)
        x = self.bottleneck(x)

        # 任务特定分支
        vp = self.decoder_vp(x)
        density = self.decoder_density(x)

        return vp.squeeze(1), density.squeeze(1)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    counter = 0

    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for seismic, vp, density in train_loader:
            seismic = seismic.to(device).permute(0, 2, 1)  # [batch, 1, 2800]
            vp = vp.to(device)  # [batch, 2800]
            density = density.to(device)  # [batch, 2800]

            optimizer.zero_grad()

            vp_pred, density_pred = model(seismic)  # 输出应为 [batch, 2800]

            # 检查尺寸是否匹配
            if vp_pred.shape != vp.shape:
                print(f"尺寸不匹配! vp_pred: {vp_pred.shape}, vp: {vp.shape}")
                continue

            loss_vp = criterion(vp_pred, vp)
            loss_density = criterion(density_pred, density)
            loss = loss_vp + loss_density

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * seismic.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seismic, vp, density in val_loader:
                seismic = seismic.to(device).permute(0, 2, 1)
                vp = vp.to(device)
                density = density.to(device)

                vp_pred, density_pred = model(seismic)

                # 检查尺寸
                if vp_pred.shape != vp.shape:
                    continue

                loss_vp = criterion(vp_pred, vp)
                loss_density = criterion(density_pred, density)
                loss = loss_vp + loss_density

                val_loss += loss.item() * seismic.size(0)

        val_loss /= len(val_loader.dataset)
        val_loss_history.append(val_loss)

        epoch_time = time.time() - epoch_start

        print(f'\nEpoch {epoch + 1}/{num_epochs} - Time: {epoch_time:.2f}s')
        print(f'Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}')

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
            print(f'Validation loss improved to {val_loss:.6f}')
        else:
            counter += 1
            print(f'Validation loss did not improve for {counter}/{patience} epochs')
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    # 加载最佳模型
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history


# 评估函数
def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss_vp = 0.0
    total_loss_density = 0.0
    total_points = 0

    with torch.no_grad():
        for seismic, vp, density in data_loader:
            seismic = seismic.to(device).permute(0, 2, 1)
            vp = vp.to(device)
            density = density.to(device)

            vp_pred, density_pred = model(seismic)

            # 跳过尺寸不匹配的批次
            if vp_pred.shape != vp.shape or density_pred.shape != density.shape:
                continue

            loss_vp = criterion(vp_pred, vp)
            loss_density = criterion(density_pred, density)

            total_loss_vp += loss_vp.item() * vp.numel()
            total_loss_density += loss_density.item() * density.numel()
            total_points += vp.numel()

    if total_points == 0:
        return float('inf'), float('inf')

    mse_vp = total_loss_vp / total_points
    mse_density = total_loss_density / total_points

    return mse_vp, mse_density


# 主函数
def main():
    print(f"Using device: {device}")

    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)

    # 加载数据
    seismic = np.load('seismic.npy').T  # 转置为(2800, 13601)
    vp = np.load('P_WAVE_VELOCITY.npy').T / 1000.0  # 转置并转换为km/s
    density = np.load('DENSITY.npy').T

    print("Data shapes:")
    print(f"Seismic: {seismic.shape}")
    print(f"Vp: {vp.shape}")
    print(f"Density: {density.shape}")

    # 数据预处理 - 标准化
    seismic_scaler = StandardScaler()
    vp_scaler = StandardScaler()
    density_scaler = StandardScaler()

    # 对每个道进行标准化
    seismic_processed = np.zeros_like(seismic)
    vp_processed = np.zeros_like(vp)
    density_processed = np.zeros_like(density)

    for i in range(seismic.shape[1]):
        seismic_processed[:, i] = seismic_scaler.fit_transform(seismic[:, i].reshape(-1, 1)).flatten()
        vp_processed[:, i] = vp_scaler.fit_transform(vp[:, i].reshape(-1, 1)).flatten()
        density_processed[:, i] = density_scaler.fit_transform(density[:, i].reshape(-1, 1)).flatten()

    # 创建数据集
    dataset = SeismicDataset(seismic_processed.T, vp_processed.T, density_processed.T)

    # 划分数据集 - 使用最少标签数据
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.95,  # 仅使用5%的数据进行训练
        random_state=42
    )

    # 从训练集中再划分验证集
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    print(f"Train samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")

    # 创建数据加载器
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # 初始化模型
    model = MultiTaskUNet(input_channels=1, output_channels=1).to(device)

    # 打印模型结构
    print("Model architecture:")
    print(model)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5#, verbose=True
                                                     )

    # 训练模型
    print("Starting training...")
    model, train_history, val_history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=100, patience=20
    )

    # 可视化训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='Train Loss')
    plt.plot(val_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/training_loss.png')
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), 'output/multitask_seismic_inversion.pth')
    print("Model saved to output/multitask_seismic_inversion.pth")

    # 可视化示例结果
    visualize_results(model, test_dataset, vp_scaler, density_scaler)


# 可视化结果函数
def visualize_results(model, dataset, vp_scaler, density_scaler):
    model.eval()
    idx = np.random.randint(0, len(dataset))
    seismic, vp_true, density_true = dataset[idx]

    with torch.no_grad():
        seismic_input = seismic.unsqueeze(0).permute(0, 2, 1).to(device)
        vp_pred, density_pred = model(seismic_input)

    # 逆标准化
    vp_true = vp_scaler.inverse_transform(vp_true.numpy().reshape(-1, 1)).flatten()
    density_true = density_scaler.inverse_transform(density_true.numpy().reshape(-1, 1)).flatten()
    vp_pred = vp_scaler.inverse_transform(vp_pred.cpu().numpy().reshape(-1, 1)).flatten()
    density_pred = density_scaler.inverse_transform(density_pred.cpu().numpy().reshape(-1, 1)).flatten()

    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    time_axis = np.linspace(0, 2.8, 2800)  # 假设2.8秒

    # Vp结果
    axes[0, 0].plot(time_axis, vp_true, 'b-', label='True Vp')
    axes[0, 0].plot(time_axis, vp_pred, 'r--', label='Predicted Vp')
    axes[0, 0].set_title('P-wave Velocity Comparison')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Vp (km/s)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 密度结果
    axes[0, 1].plot(time_axis, density_true, 'b-', label='True Density')
    axes[0, 1].plot(time_axis, density_pred, 'r--', label='Predicted Density')
    axes[0, 1].set_title('Density Comparison')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Density (g/cm³)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Vp误差
    vp_error = vp_pred - vp_true
    axes[1, 0].plot(time_axis, vp_error, 'g-')
    axes[1, 0].set_title('Vp Prediction Error')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Error (km/s)')
    axes[1, 0].grid(True)

    # 密度误差
    density_error = density_pred - density_true
    axes[1, 1].plot(time_axis, density_error, 'g-')
    axes[1, 1].set_title('Density Prediction Error')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Error (g/cm³)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('output/inversion_results.png')
    plt.show()


if __name__ == "__main__":
    main()