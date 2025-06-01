import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import os

# 设置全局绘图参数（学术级质量）
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# 创建输出目录
os.makedirs("draw_images", exist_ok=True)


def plot_seismic_data():
    """绘制地震数据剖面图"""
    seismic = np.load('seismic.npy').T  # 转置为(2800, 13601)

    # 创建网格
    X = np.arange(seismic.shape[1])  # 道号 0-13600
    Y = np.linspace(0, 2.799, seismic.shape[0])  # 时间 0-2.799秒

    fig, ax = plt.subplots(figsize=(15, 8))

    # 使用pcolormesh代替contourf以提高性能
    im = ax.pcolormesh(X, Y, seismic, cmap='seismic', shading='auto', vmin=-np.max(np.abs(seismic)),
                       vmax=np.max(np.abs(seismic)))

    ax.set_xlabel("Trace Number", fontweight='bold')
    ax.set_ylabel("Time (s)", fontweight='bold')
    ax.set_title("Seismic Data Profile", fontsize=18, fontweight='bold')

    # 反转时间轴（0在顶部）
    ax.invert_yaxis()

    # 添加色标
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Amplitude", fontweight='bold')

    plt.savefig("draw_images/seismic_data.png", dpi=300)
    plt.close()
    print("Successfully saved seismic_data.png")


def plot_vp_density():
    """绘制P波速度和密度真实值对比图"""
    vp = np.load('P_WAVE_VELOCITY.npy').T / 1000.0  # 转换为km/s
    density = np.load('DENSITY.npy').T

    # 创建网格
    X = np.arange(vp.shape[1])  # 道号 0-13600
    Y = np.linspace(0, 2.799, vp.shape[0])  # 时间 0-2.799秒

    fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    # 绘制P波速度
    im1 = axs[0].pcolormesh(X, Y, vp, cmap='viridis', shading='auto', vmin=1.0, vmax=4.7)
    axs[0].set_ylabel("Time (s)", fontweight='bold')
    axs[0].set_title("P-wave Velocity (True)", fontsize=16, fontweight='bold')
    axs[0].invert_yaxis()

    # 添加色标
    cbar1 = fig.colorbar(im1, ax=axs[0], shrink=0.7)
    cbar1.set_label("Velocity (km/s)", fontweight='bold')

    # 绘制密度
    im2 = axs[1].pcolormesh(X, Y, density, cmap='plasma', shading='auto', vmin=1.5, vmax=3.0)
    axs[1].set_xlabel("Trace Number", fontweight='bold')
    axs[1].set_ylabel("Time (s)", fontweight='bold')
    axs[1].set_title("Density (True)", fontsize=16, fontweight='bold')
    axs[1].invert_yaxis()

    # 添加色标
    cbar2 = fig.colorbar(im2, ax=axs[1], shrink=0.7)
    cbar2.set_label("Density (g/cm³)", fontweight='bold')

    plt.tight_layout()
    plt.savefig("draw_images/vp_density.png", dpi=300)
    plt.close()
    print("Successfully saved vp_density.png")


def plot_vp_results():
    """绘制P波速度真实值与预测值对比"""
    # 加载真实值和预测值
    vp_true = np.load('P_WAVE_VELOCITY.npy').T / 1000.0  # 转换为km/s
    vp_pred = np.load('evaluate_results/vp_predictions.npy').T

    # 创建网格
    X = np.arange(vp_true.shape[1])  # 道号 0-13600
    Y = np.linspace(0, 2.799, vp_true.shape[0])  # 时间 0-2.799秒

    # 创建2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True)

    # 绘制真实P波速度
    im_true = axs[0, 0].pcolormesh(X, Y, vp_true, cmap='viridis', shading='auto', vmin=1.0, vmax=4.7)
    axs[0, 0].set_ylabel("Time (s)", fontweight='bold')
    axs[0, 0].set_title("True P-wave Velocity", fontsize=16, fontweight='bold')
    axs[0, 0].invert_yaxis()

    # 添加色标
    cbar_true = fig.colorbar(im_true, ax=axs[0, 0], shrink=0.7)
    cbar_true.set_label("Velocity (km/s)", fontweight='bold')

    # 绘制预测P波速度
    im_pred = axs[0, 1].pcolormesh(X, Y, vp_pred, cmap='viridis', shading='auto', vmin=1.0, vmax=4.7)
    axs[0, 1].set_title("Predicted P-wave Velocity", fontsize=16, fontweight='bold')
    axs[0, 1].invert_yaxis()

    # 添加色标
    cbar_pred = fig.colorbar(im_pred, ax=axs[0, 1], shrink=0.7)
    cbar_pred.set_label("Velocity (km/s)", fontweight='bold')

    # 单独保存真实值和预测值图像（用于Beamer分开展示）
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.pcolormesh(X, Y, vp_true, cmap='viridis', shading='auto', vmin=1.0, vmax=4.7)
    ax.set_xlabel("Trace Number", fontweight='bold')
    ax.set_ylabel("Time (s)", fontweight='bold')
    ax.set_title("True P-wave Velocity", fontsize=18, fontweight='bold')
    ax.invert_yaxis()
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Velocity (km/s)", fontweight='bold')
    plt.savefig("draw_images/vp_true.png", dpi=300)
    plt.close()
    print("Successfully saved vp_true.png")

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.pcolormesh(X, Y, vp_pred, cmap='viridis', shading='auto', vmin=1.0, vmax=4.7)
    ax.set_xlabel("Trace Number", fontweight='bold')
    ax.set_ylabel("Time (s)", fontweight='bold')
    ax.set_title("Predicted P-wave Velocity", fontsize=18, fontweight='bold')
    ax.invert_yaxis()
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Velocity (km/s)", fontweight='bold')
    plt.savefig("draw_images/vp_pred.png", dpi=300)
    plt.close()
    print("Successfully saved vp_pred.png")


def plot_density_results():
    """绘制密度真实值与预测值对比"""
    # 加载真实值和预测值
    density_true = np.load('DENSITY.npy').T
    density_pred = np.load('evaluate_results/density_predictions.npy').T

    # 创建网格
    X = np.arange(density_true.shape[1])  # 道号 0-13600
    Y = np.linspace(0, 2.799, density_true.shape[0])  # 时间 0-2.799秒

    # 创建2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True)

    # 绘制真实密度
    im_true = axs[0, 0].pcolormesh(X, Y, density_true, cmap='plasma', shading='auto', vmin=1.5, vmax=3.0)
    axs[0, 0].set_ylabel("Time (s)", fontweight='bold')
    axs[0, 0].set_title("True Density", fontsize=16, fontweight='bold')
    axs[0, 0].invert_yaxis()

    # 添加色标
    cbar_true = fig.colorbar(im_true, ax=axs[0, 0], shrink=0.7)
    cbar_true.set_label("Density (g/cm³)", fontweight='bold')

    # 绘制预测密度
    im_pred = axs[0, 1].pcolormesh(X, Y, density_pred, cmap='plasma', shading='auto', vmin=1.5, vmax=3.0)
    axs[0, 1].set_title("Predicted Density", fontsize=16, fontweight='bold')
    axs[0, 1].invert_yaxis()

    # 添加色标
    cbar_pred = fig.colorbar(im_pred, ax=axs[0, 1], shrink=0.7)
    cbar_pred.set_label("Density (g/cm³)", fontweight='bold')

    # 单独保存真实值和预测值图像（用于Beamer分开展示）
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.pcolormesh(X, Y, density_true, cmap='plasma', shading='auto', vmin=1.5, vmax=3.0)
    ax.set_xlabel("Trace Number", fontweight='bold')
    ax.set_ylabel("Time (s)", fontweight='bold')
    ax.set_title("True Density", fontsize=18, fontweight='bold')
    ax.invert_yaxis()
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Density (g/cm³)", fontweight='bold')
    plt.savefig("draw_images/density_true.png", dpi=300)
    plt.close()
    print("Successfully saved density_true.png")

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.pcolormesh(X, Y, density_pred, cmap='plasma', shading='auto', vmin=1.5, vmax=3.0)
    ax.set_xlabel("Trace Number", fontweight='bold')
    ax.set_ylabel("Time (s)", fontweight='bold')
    ax.set_title("Predicted Density", fontsize=18, fontweight='bold')
    ax.invert_yaxis()
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Density (g/cm³)", fontweight='bold')
    plt.savefig("draw_images/density_pred.png", dpi=300)
    plt.close()
    print("Successfully saved density_pred.png")


def plot_vp_comparision():
    """绘制P波速度真实值与预测值对比"""
    try:
        # 加载真实值和预测值 - 确保正确转置
        vp_true = np.load('P_WAVE_VELOCITY.npy') / 1000.0  # (13601, 2800)
        vp_pred = np.load('evaluate_results/vp_predictions.npy')  # (13601, 2800)

        print(f"vp_true shape: {vp_true.shape}, min: {np.min(vp_true):.2f}, max: {np.max(vp_true):.2f}")
        print(f"vp_pred shape: {vp_pred.shape}, min: {np.min(vp_pred):.2f}, max: {np.max(vp_pred):.2f}")

        # 创建网格 - 确保与数据维度匹配
        trace_num = vp_true.shape[0]  # 13601
        time_points = vp_true.shape[1]  # 2800
        X = np.arange(trace_num)  # 道号 0-13600
        Y = np.linspace(0, 2.799, time_points)  # 时间 0-2.799秒

        # 创建2x2的子图布局
        fig, axs = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle("P-wave Velocity Inversion Results", fontsize=20, fontweight='bold')

        # 绘制真实P波速度
        im_true = axs[0, 0].imshow(vp_true.T,  # 转置为(2800, 13601)
                                   aspect='auto',
                                   cmap='viridis',
                                   extent=[0, trace_num, 2.799, 0],  # [xmin, xmax, ymin, ymax]
                                   vmin=1.0,
                                   vmax=4.7)
        axs[0, 0].set_ylabel("Time (s)", fontweight='bold')
        axs[0, 0].set_title("True P-wave Velocity", fontsize=16, fontweight='bold')
        axs[0, 0].set_xlabel("Trace Number", fontweight='bold')
        fig.colorbar(im_true, ax=axs[0, 0], label="Velocity (km/s)")

        # 绘制预测P波速度
        im_pred = axs[0, 1].imshow(vp_pred.T,  # 转置为(2800, 13601)
                                   aspect='auto',
                                   cmap='viridis',
                                   extent=[0, trace_num, 2.799, 0],
                                   vmin=1.0,
                                   vmax=4.7)
        axs[0, 1].set_title("Predicted P-wave Velocity", fontsize=16, fontweight='bold')
        axs[0, 1].set_xlabel("Trace Number", fontweight='bold')
        fig.colorbar(im_pred, ax=axs[0, 1], label="Velocity (km/s)")

        # 绘制误差
        vp_error = vp_pred - vp_true
        im_error = axs[1, 0].imshow(vp_error.T,  # 转置为(2800, 13601)
                                    aspect='auto',
                                    cmap='coolwarm',
                                    extent=[0, trace_num, 2.799, 0],
                                    vmin=-0.5,
                                    vmax=0.5)
        axs[1, 0].set_xlabel("Trace Number", fontweight='bold')
        axs[1, 0].set_ylabel("Time (s)", fontweight='bold')
        axs[1, 0].set_title("Prediction Error", fontsize=16, fontweight='bold')
        fig.colorbar(im_error, ax=axs[1, 0], label="Error (km/s)")

        # 绘制误差直方图
        axs[1, 1].hist(vp_error.flatten(), bins=100, color='purple', alpha=0.7)
        axs[1, 1].set_xlabel("Error (km/s)", fontweight='bold')
        axs[1, 1].set_ylabel("Frequency", fontweight='bold')
        axs[1, 1].set_title("Error Distribution", fontsize=16, fontweight='bold')
        axs[1, 1].grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = (f"Mean Error: {np.mean(vp_error):.4f} km/s\n"
                      f"Std Dev: {np.std(vp_error):.4f} km/s\n"
                      f"MSE: {np.mean(vp_error ** 2):.6f}")
        axs[1, 1].text(0.05, 0.95, stats_text, transform=axs[1, 1].transAxes,
                       fontsize=14, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig("draw_images/vp_comparison.png", dpi=300)
        plt.close()
        print("Successfully saved vp_comparison.png")

    except Exception as e:
        print(f"Error in plot_vp_results: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_density_comparision():
    """绘制密度真实值与预测值对比"""
    try:
        # 加载真实值和预测值 - 确保正确转置
        density_true = np.load('DENSITY.npy')  # (13601, 2800)
        density_pred = np.load('evaluate_results/density_predictions.npy')  # (13601, 2800)

        print(
            f"density_true shape: {density_true.shape}, min: {np.min(density_true):.2f}, max: {np.max(density_true):.2f}")
        print(
            f"density_pred shape: {density_pred.shape}, min: {np.min(density_pred):.2f}, max: {np.max(density_pred):.2f}")

        # 创建网格 - 确保与数据维度匹配
        trace_num = density_true.shape[0]  # 13601
        time_points = density_true.shape[1]  # 2800
        X = np.arange(trace_num)  # 道号 0-13600
        Y = np.linspace(0, 2.799, time_points)  # 时间 0-2.799秒

        # 创建2x2的子图布局
        fig, axs = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle("Density Inversion Results", fontsize=20, fontweight='bold')

        # 绘制真实密度
        im_true = axs[0, 0].imshow(density_true.T,  # 转置为(2800, 13601)
                                   aspect='auto',
                                   cmap='plasma',
                                   extent=[0, trace_num, 2.799, 0],
                                   vmin=1.5,
                                   vmax=3.0)
        axs[0, 0].set_ylabel("Time (s)", fontweight='bold')
        axs[0, 0].set_title("True Density", fontsize=16, fontweight='bold')
        axs[0, 0].set_xlabel("Trace Number", fontweight='bold')
        fig.colorbar(im_true, ax=axs[0, 0], label="Density (g/cm³)")

        # 绘制预测密度
        im_pred = axs[0, 1].imshow(density_pred.T,  # 转置为(2800, 13601)
                                   aspect='auto',
                                   cmap='plasma',
                                   extent=[0, trace_num, 2.799, 0],
                                   vmin=1.5,
                                   vmax=3.0)
        axs[0, 1].set_title("Predicted Density", fontsize=16, fontweight='bold')
        axs[0, 1].set_xlabel("Trace Number", fontweight='bold')
        fig.colorbar(im_pred, ax=axs[0, 1], label="Density (g/cm³)")

        # 绘制误差
        density_error = density_pred - density_true
        im_error = axs[1, 0].imshow(density_error.T,  # 转置为(2800, 13601)
                                    aspect='auto',
                                    cmap='coolwarm',
                                    extent=[0, trace_num, 2.799, 0],
                                    vmin=-0.1,
                                    vmax=0.1)
        axs[1, 0].set_xlabel("Trace Number", fontweight='bold')
        axs[1, 0].set_ylabel("Time (s)", fontweight='bold')
        axs[1, 0].set_title("Prediction Error", fontsize=16, fontweight='bold')
        fig.colorbar(im_error, ax=axs[1, 0], label="Error (g/cm³)")

        # 绘制误差直方图
        axs[1, 1].hist(density_error.flatten(), bins=100, color='green', alpha=0.7)
        axs[1, 1].set_xlabel("Error (g/cm³)", fontweight='bold')
        axs[1, 1].set_ylabel("Frequency", fontweight='bold')
        axs[1, 1].set_title("Error Distribution", fontsize=16, fontweight='bold')
        axs[1, 1].grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = (f"Mean Error: {np.mean(density_error):.4f} g/cm³\n"
                      f"Std Dev: {np.std(density_error):.4f} g/cm³\n"
                      f"MSE: {np.mean(density_error ** 2):.6f}")
        axs[1, 1].text(0.05, 0.95, stats_text, transform=axs[1, 1].transAxes,
                       fontsize=14, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig("draw_images/density_comparison.png", dpi=300)
        plt.close()
        print("Successfully saved density_comparison.png")

    except Exception as e:
        print(f"Error in plot_density_results: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    # 生成所有需要的图像
    plot_seismic_data()
    plot_vp_density()
    plot_vp_results()
    plot_density_results()
    plot_vp_comparision()
    plot_density_comparision()

    print("\nAll images saved to 'draw_images' directory")


if __name__ == "__main__":
    main()