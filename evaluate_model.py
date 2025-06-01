import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import time
import matplotlib.pyplot as plt
import argparse

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义与训练时相同的模型架构
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

        # 任务特定解码器 - Vp
        self.decoder_vp = nn.Sequential(
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

        # 任务特定解码器 - Density
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

    def forward(self, x):
        # 共享特征提取
        x = self.encoder(x)
        x = self.bottleneck(x)

        # 任务特定分支
        vp = self.decoder_vp(x)
        density = self.decoder_density(x)

        return vp.squeeze(1), density.squeeze(1)


# 自定义数据集类（用于批量处理）
class InferenceDataset(Dataset):
    def __init__(self, seismic, vp, density):
        self.seismic = seismic
        self.vp = vp
        self.density = density
        self.seismic_stats = []
        self.vp_stats = []
        self.density_stats = []

        # 预计算每个道的统计信息
        for i in range(seismic.shape[1]):
            self.seismic_stats.append((np.mean(seismic[:, i]), np.std(seismic[:, i])))
            self.vp_stats.append((np.mean(vp[:, i]), np.std(vp[:, i])))
            self.density_stats.append((np.mean(density[:, i]), np.std(density[:, i])))

    def __len__(self):
        return self.seismic.shape[1]  # 每个道作为一个样本

    def __getitem__(self, idx):
        # 获取当前道
        seismic_trace = self.seismic[:, idx].reshape(-1).astype(np.float32)
        vp_trace = self.vp[:, idx].reshape(-1).astype(np.float32)
        density_trace = self.density[:, idx].reshape(-1).astype(np.float32)

        # 标准化地震道
        mean, std = self.seismic_stats[idx]
        seismic_trace = (seismic_trace - mean) / (std + 1e-8)

        # 直接返回统计值而不是元组
        vp_mean, vp_std = self.vp_stats[idx]
        density_mean, density_std = self.density_stats[idx]

        return (
            torch.tensor(seismic_trace),
            torch.tensor(vp_trace),
            torch.tensor(density_trace),
            idx,
            vp_mean,  # 直接返回均值
            vp_std,  # 直接返回标准差
            density_mean,
            density_std
        )


# 计算整个数据集的MSE
def calculate_total_mse(model, data_loader):
    model.eval()

    # 创建数组保存所有预测结果
    vp_pred_all = np.zeros((13601, 2800))
    density_pred_all = np.zeros((13601, 2800))
    vp_true_all = np.zeros((13601, 2800))
    density_true_all = np.zeros((13601, 2800))

    start_time = time.time()
    processed_batches = 0
    total_batches = len(data_loader)

    with torch.no_grad():
        for batch_idx, (seismic, vp_true, density_true, indices, vp_means, vp_stds, density_means,
                        density_stds) in enumerate(data_loader):
            # 移动到设备并确保数据类型为float32
            seismic = seismic.to(device).unsqueeze(1).float()  # 添加通道维度并转换为float32 [batch, 1, 2800]

            # 预测
            vp_pred, density_pred = model(seismic)

            # 转换为numpy数组
            vp_pred = vp_pred.cpu().numpy()
            density_pred = density_pred.cpu().numpy()
            vp_true_batch = vp_true.cpu().numpy()
            density_true_batch = density_true.cpu().numpy()

            # 逆标准化预测结果
            for i in range(len(indices)):
                idx = indices[i].item()

                # 获取当前道的统计信息
                vp_mean = vp_means[i].item()
                vp_std = vp_stds[i].item()
                density_mean = density_means[i].item()
                density_std = density_stds[i].item()

                vp_pred_i = vp_pred[i] * vp_std + vp_mean
                density_pred_i = density_pred[i] * density_std + density_mean

                # 保存结果
                vp_pred_all[idx] = vp_pred_i
                density_pred_all[idx] = density_pred_i
                vp_true_all[idx] = vp_true_batch[i]
                density_true_all[idx] = density_true_batch[i]

            # 打印进度
            processed_batches = batch_idx + 1
            if processed_batches % 100 == 0 or processed_batches == total_batches:
                elapsed = time.time() - start_time
                print(
                    f'Processed {processed_batches}/{total_batches} batches ({100 * processed_batches / total_batches:.1f}%), '
                    f'Elapsed: {elapsed:.1f}s')

    # 计算整个数据集的MSE
    mse_vp = np.mean((vp_pred_all - vp_true_all) ** 2)
    mse_density = np.mean((density_pred_all - density_true_all) ** 2)

    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.1f} seconds")

    return mse_vp, mse_density, vp_pred_all, density_pred_all, vp_true_all, density_true_all


# 可视化部分结果
def visualize_results(vp_pred, vp_true, density_pred, density_true, n_traces=5, save_dir="output"):
    os.makedirs(save_dir, exist_ok=True)

    # 随机选择一些道进行可视化
    trace_indices = np.random.choice(vp_pred.shape[0], n_traces, replace=False)

    for i, idx in enumerate(trace_indices):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        time_axis = np.linspace(0, 2.8, 2800)  # 假设2.8秒

        # Vp结果
        axs[0, 0].plot(time_axis, vp_true[idx], 'b-', label='True Vp')
        axs[0, 0].plot(time_axis, vp_pred[idx], 'r--', label='Predicted Vp')
        axs[0, 0].set_title(f'P-wave Velocity Comparison (Trace {idx})')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Vp (km/s)')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # 密度结果
        axs[0, 1].plot(time_axis, density_true[idx], 'b-', label='True Density')
        axs[0, 1].plot(time_axis, density_pred[idx], 'r--', label='Predicted Density')
        axs[0, 1].set_title(f'Density Comparison (Trace {idx})')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Density (g/cm³)')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Vp误差
        vp_error = vp_pred[idx] - vp_true[idx]
        axs[1, 0].plot(time_axis, vp_error, 'g-')
        axs[1, 0].set_title(f'Vp Prediction Error (Trace {idx})')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Error (km/s)')
        axs[1, 0].grid(True)

        # 密度误差
        density_error = density_pred[idx] - density_true[idx]
        axs[1, 1].plot(time_axis, density_error, 'g-')
        axs[1, 1].set_title(f'Density Prediction Error (Trace {idx})')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Error (g/cm³)')
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/trace_{idx}_results.png")
        plt.close(fig)

    print(f"Saved visualization for {n_traces} traces in {save_dir}")


def main(model_path, batch_size=16, visualize=True):
    print(f"Using device: {device}")

    # 1. 加载数据
    print("Loading data...")
    seismic = np.load('seismic.npy').T  # 转置为(2800, 13601)
    vp = np.load('P_WAVE_VELOCITY.npy').T / 1000.0  # 转置并转换为km/s
    density = np.load('DENSITY.npy').T

    # 确保数据类型为float32
    seismic = seismic.astype(np.float32)
    vp = vp.astype(np.float32)
    density = density.astype(np.float32)

    print("Data shapes:")
    print(f"Seismic: {seismic.shape}")
    print(f"Vp: {vp.shape}")
    print(f"Density: {density.shape}")

    # 2. 创建数据集和数据加载器
    dataset = InferenceDataset(seismic, vp, density)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. 加载模型
    print(f"Loading model from {model_path}...")
    model = MultiTaskUNet(input_channels=1, output_channels=1).to(device)
    model = model.float()  # 确保模型使用float32
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. 计算整个数据集的MSE
    print("\nStarting inference on entire dataset...")
    mse_vp, mse_density, vp_pred, density_pred, vp_true, density_true = calculate_total_mse(model, data_loader)

    # 5. 输出结果
    print("\n" + "=" * 50)
    print("Final MSE Results:")
    print(f"P-wave Velocity MSE: {mse_vp:.6f} (km/s)^2")
    print(f"Density MSE: {mse_density:.6f} (g/cm³)^2")
    print("=" * 50)

    # 6. 保存结果
    os.makedirs('evaluate_results', exist_ok=True)

    with open('evaluate_results/evaluate_MSE.txt', 'w', encoding='utf-8') as file:
        print("=" * 50 + "\n", file=file)
        print("Final MSE Results:\n", file=file)
        print(f"P-wave Velocity MSE: {mse_vp:.6f} (km/s)^2\n", file=file)
        print(f"Density MSE: {mse_density:.6f} (g/cm³)^2\n", file=file)
        print("=" * 50, file=file)

    np.save("evaluate_results/vp_predictions.npy", vp_pred)
    np.save("evaluate_results/density_predictions.npy", density_pred)
    np.save("evaluate_results/vp_true.npy", vp_true)
    np.save("evaluate_results/density_true.npy", density_true)
    print("\nPredictions saved to disk.")

    # 7. 可视化部分结果
    if visualize:
        print("\nVisualizing sample results...")
        visualize_results(vp_pred, vp_true, density_pred, density_true, save_dir='evaluate_results')


if __name__ == "__main__":

    main(
        model_path='output/multitask_seismic_inversion.pth',
        batch_size=32,
        visualize=True
    )