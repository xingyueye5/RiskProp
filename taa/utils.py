import os
import cv2
import numpy as np
import torch  # 假设 Tensor 是 PyTorch 的格式
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import pandas as pd

# 可视化配置参数
VISUALIZATION_CONFIG = {
    'font_family': ['Times New Roman', 'DejaVu Sans'],
    'font_size': 12,
    'line_width': 2.5,
    'colors': {
        'prediction_line': '#2E86AB',      # 深蓝色
        'threshold_line': '#A23B72',       # 紫色
        'accident_line': '#C73E1D',        # 橙色
        'abnormal_line': '#F18F01',        # 红色
        'histogram': '#2E86AB',            # 深蓝色
    },
    'figure_dpi': 300,
    'save_format': 'png',
    'grid_alpha': 0.3,
    'legend_alpha': 0.9,
}


def get_fps(video_id, dataset_type="cap"):
    assert dataset_type in ["cap", "dada"]  # 仅支持 'cap' 和 'dada' 数据集
    if dataset_type == "cap":
        if "000001" <= video_id <= "006381":
            return 10
        elif "006382" <= video_id <= "007887":
            return 30
        elif "007888" <= video_id <= "009046":
            return 20
        elif "009047" <= video_id <= "011770":
            return 30
        elif "0130001" <= video_id <= "014490":
            return 10
        else:
            return None
    else:  # dataset_type == 'dada'
        return 30


def visualize_tensor_as_videos(tensor, output_dir, fps=10):
    """
    将形状为 NCTHW 的 tensor 可视化为 N 个视频，存储到指定文件夹中。

    :param tensor: torch.Tensor, 形状为 N x C x T x H x W
    :param output_dir: str, 保存视频的文件夹路径
    :param fps: int, 视频帧率
    """
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取已有视频文件数量
    existing_videos = [f for f in os.listdir(output_dir) if f.endswith(".mp4") and f.startswith("video_")]
    start_index_video = len(existing_videos)
    existing_clips = [f for f in os.listdir(output_dir) if f.endswith(".mp4") and f.startswith("clip_")]
    start_index_clip = len(existing_clips)

    # 将 Tensor 转换为 NumPy 数组
    tensor = tensor.cpu().numpy()  # 假设 Tensor 在 GPU 上
    if len(tensor.shape) == 4:
        N, C, H, W = tensor.shape
        # 创建视频保存路径
        video_path = os.path.join(output_dir, f"video_{start_index_video}.mp4")

        # 创建 VideoWriter 对象
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 MP4 编码
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

        for t in range(N):
            frame = tensor[t, :, :, :]  # 提取单帧 (C x H x W)
            if C == 1:  # 灰度图
                frame = np.squeeze(frame, axis=0)  # 去掉通道维度
                frame = frame.astype(np.uint8)  # 标准化到 [0, 255]
            elif C == 3:  # RGB
                frame = frame.transpose(1, 2, 0)  # 变换为 H x W x C
                frame = frame.astype(np.uint8)
            else:
                raise ValueError(f"不支持的通道数：{C}")

            # 写入帧到视频
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 转换为 BGR 格式

        # 释放 VideoWriter
        video_writer.release()

    elif len(tensor.shape) == 5:
        N, C, T, H, W = tensor.shape
        for i in range(N):
            # 创建视频保存路径
            video_path = os.path.join(output_dir, f"clip_{start_index_clip + i}.mp4")

            # 创建 VideoWriter 对象
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 MP4 编码
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

            for t in range(T):
                frame = tensor[i, :, t, :, :]  # 提取单帧 (C x H x W)
                if C == 1:  # 灰度图
                    frame = np.squeeze(frame, axis=0)  # 去掉通道维度
                    frame = frame.astype(np.uint8)  # 标准化到 [0, 255]
                elif C == 3:  # RGB
                    frame = frame.transpose(1, 2, 0)  # 变换为 H x W x C
                    frame = frame.astype(np.uint8)
                else:
                    raise ValueError(f"不支持的通道数：{C}")

                # 写入帧到视频
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 转换为 BGR 格式

            # 释放 VideoWriter
            video_writer.release()

    print(f"所有视频已保存到：{output_dir}")



## 可视化预测分数曲线,都加一个偏移
def visualize_pred_line(result, output_dir, epoch=None, fps=10):
    pred = result["pred"]
    target = result["target"]
    accident_ind = result["accident_ind"]
    video_id = result["video_id"]
    dataset = result["dataset"]
    frame_dir = result["frame_dir"]
    filename_tmpl = result["filename_tmpl"]
    type = result["type"]
    frame_inds = result["frame_inds"]
    abnormal_start_frame = result["abnormal_start_frame"]
    accident_frame = result["accident_frame"]
    threshold = result["threshold"]

    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体和样式，符合顶级会议论文格式
    plt.rcParams['font.family'] = VISUALIZATION_CONFIG['font_family']
    plt.rcParams['font.size'] = VISUALIZATION_CONFIG['font_size']
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # 创建高质量的图形
    fig, ax = plt.subplots(figsize=(10, 6), dpi=VISUALIZATION_CONFIG['figure_dpi'])
    
    # 绘制预测分数曲线
    line_color = VISUALIZATION_CONFIG['colors']['prediction_line']
    line_width = VISUALIZATION_CONFIG['line_width']
    ax.plot(frame_inds-frame_inds[0], pred, color=line_color, linewidth=line_width, 
            label='Prediction Score', alpha=0.9, zorder=3)
    
    # 绘制阈值线
    threshold_color = VISUALIZATION_CONFIG['colors']['threshold_line']
    ax.axhline(y=threshold, color=threshold_color, linestyle='--', 
               linewidth=2, alpha=0.25, label=f'Threshold ({threshold:.2f})', zorder=2)

    
    # 在accident frame位置画红色竖直虚线
    if accident_frame is not None :
        accident_color = VISUALIZATION_CONFIG['colors']['accident_line']
        ax.axvline(x=accident_frame, color=accident_color, linestyle='--', 
                   linewidth=2.5, alpha=0.5, label='Accident Frame', zorder=4)
        
        # 添加事故帧的标注
        ax.annotate('Accident', xy=(accident_frame, ax.get_ylim()[1]), 
                   xytext=(accident_frame + 5, ax.get_ylim()[1] * 0.97),
                   arrowprops=dict(arrowstyle='->', color=accident_color, lw=1.5),
                   fontsize=10, color=accident_color, weight='bold')
    
    # 在abnormal start frame位置画竖直虚线
    if abnormal_start_frame is not None :
        abnormal_color = VISUALIZATION_CONFIG['colors']['abnormal_line']
        ax.axvline(x=abnormal_start_frame-frame_inds[0], color=abnormal_color, linestyle=':', 
                   linewidth=2, alpha=0.5, label='Abnormal Start', zorder=4)
        
        # 添加异常开始帧的标注
        ax.annotate('Abnormal', xy=(abnormal_start_frame-frame_inds[0], ax.get_ylim()[0]), 
                   xytext=(abnormal_start_frame-frame_inds[0] - 5, ax.get_ylim()[0] * 0.97),
                   arrowprops=dict(arrowstyle='->', color=abnormal_color, lw=1.5),
                   fontsize=9, color=abnormal_color, weight='bold',
                   ha='center', va='bottom')
    
    # 设置图表属性
    #ax.set_xlabel('Frame Index', fontsize=14, weight='bold', fontfamily='Times New Roman')
    ax.set_xlabel('Average Anticipation Score Evolution on Nexar Dataset', fontsize=14, weight='bold', fontfamily='Times New Roman')
    ax.set_ylabel('Prediction Score', fontsize=14, weight='bold', fontfamily='Times New Roman')
    
    
    # 设置网格
    ax.grid(True, alpha=VISUALIZATION_CONFIG['grid_alpha'], linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # 设置坐标轴范围
    ax.set_xlim(0, max(frame_inds))
    ax.set_ylim(0, max(1.0, max(pred) * 1.1))
    
    # 设置刻度标签
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=6)
    ax.tick_params(axis='both', which='minor', width=0.8, length=3)
    
    # 添加图例
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, 
                      shadow=True, fontsize=11, framealpha=VISUALIZATION_CONFIG['legend_alpha'])
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 根据数据集和视频ID命名图片
    epoch_suffix = f"_{epoch}" if epoch is not None else ""
    if dataset == "cap":
        plot_filename = f"{dataset}_{type}_{video_id}{epoch_suffix}_pred_score.png"
    elif dataset == "nexar":
        plot_filename = f"{dataset}_{video_id}{epoch_suffix}_pred_score.png"
    else:
        plot_filename = f"{dataset}_{video_id}{epoch_suffix}_pred_score.png"
    
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"预测分数曲线已保存到: {plot_path}")



def visualize_pred_score(result, output_dir, epoch=None, fps=10):
    pred = result["pred"]
    target = result["target"]
    accident_ind = result["accident_ind"]
    video_id = result["video_id"]
    dataset = result["dataset"]
    frame_dir = result["frame_dir"]
    filename_tmpl = result["filename_tmpl"]
    type = result["type"]
    frame_inds = result["frame_inds"]
    abnormal_start_frame = result["abnormal_start_frame"]
    accident_frame = result["accident_frame"]
    threshold = result["threshold"]

    os.makedirs(output_dir, exist_ok=True)
    # 创建视频保存路径
    epoch = f"_{epoch}" if epoch is not None else ""
    if dataset == "cap":
        video_path = os.path.join(output_dir, f"{dataset}_{type}_{video_id}{epoch}.mp4")
    else:
        video_path = os.path.join(output_dir, f"{dataset}_{video_id}{epoch}.mp4")

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 MP4 编码
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (640, 360))  # 使用低分辨率确保传输效率

    for i, frame_ind in enumerate(frame_inds):
        frame_path = os.path.join(frame_dir, filename_tmpl.format(frame_ind))
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (640, 360))
        if target is not None:
            if target:
                if frame_ind < abnormal_start_frame:
                    # 绘制绿色条带
                    cv2.rectangle(frame, (0, 0), (640, 36), (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        "Safe Scenario",
                        (420, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (255, 255, 255),
                        2,
                    )
                elif abnormal_start_frame <= frame_ind < accident_frame:
                    # 绘制蓝色条带
                    cv2.rectangle(frame, (0, 0), (640, 36), (255, 0, 0), -1)
                    cv2.putText(
                        frame,
                        "Anomaly Appeared",
                        (420, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (255, 255, 255),
                        2,
                    )
                elif accident_frame <= frame_ind:
                    # 绘制红色条带
                    cv2.rectangle(frame, (0, 0), (640, 36), (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        "Accident Occurred",
                        (420, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (255, 255, 255),
                        2,
                    )
            else:
                # 绘制绿色条带
                cv2.rectangle(frame, (0, 0), (640, 36), (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    "Safe Scenario",
                    (420, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                )
        # 绘制预测分数
        if len(pred.shape) == 1:
            if pred[i] >= threshold:
                cv2.rectangle(frame, (0, 0), (640, 360), (0, 0, 255), 5)
            cv2.putText(
                frame,
                f"Score: {pred[i]:.2f}  " + "*" * int(pred[i] * 10 // 1),
                (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
            )
        if len(pred.shape) == 2:
            if np.max(pred[i]) >= threshold:
                cv2.rectangle(frame, (0, 0), (640, 360), (0, 0, 255), 5)
            cv2.putText(
                frame,
                f"Score: {np.max(pred[i]):.2f}  " + "*" * int(np.max(pred[i]) * 10 // 1),
                (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Time. Pred  L  Visualization",
                (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Time. Pred  L  Visualization",
                (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )
            for j in range(len(pred[i])):
                l = int(i + j == accident_ind) if target else 0
                cv2.putText(
                    frame,
                    f"{j/fps:.1f}s  {pred[i][j]:.2f}  {l}  " + "*" * int(pred[i][j] * 10 // 1),
                    (20, 64 + 12 * j),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{j/fps:.1f}s  {pred[i][j]:.2f}  {l}  " + "*" * int(pred[i][j] * 10 // 1),
                    (20, 64 + 12 * j),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {video_path}")



def _epoch_suffix(epoch):
    return f"_{epoch}" if epoch is not None else ""


def plot_roc_curves_from_csvs(output_dir="outputs", epoch=None):
    os.makedirs(output_dir, exist_ok=True)

    # Configure styles
    plt.rcParams['font.family'] = VISUALIZATION_CONFIG['font_family']
    plt.rcParams['font.size'] = VISUALIZATION_CONFIG['font_size']
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    fig, ax = plt.subplots(figsize=(8, 6), dpi=VISUALIZATION_CONFIG['figure_dpi'])

    suffix = _epoch_suffix(epoch)
    file_map = {
        f"fpr_tpr_0{suffix}.csv": "0.0s",
        f"fpr_tpr_5{suffix}.csv": "0.5s",
        f"fpr_tpr_10{suffix}.csv": "1.0s",
        f"fpr_tpr_15{suffix}.csv": "1.5s",
    }

    any_plotted = False
    for fname, label in file_map.items():
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            continue
        data = np.genfromtxt(path, delimiter=',', names=True)
        fpr = data['fpr']
        tpr = data['tpr']
        ax.plot(fpr, tpr, linewidth=VISUALIZATION_CONFIG['line_width'], label=f"ROC ({label})")
        any_plotted = True

    if not any_plotted:
        return

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Chance')
    ax.set_xlabel('False Positive Rate', fontsize=14, weight='bold', fontfamily='Times New Roman')
    ax.set_ylabel('True Positive Rate', fontsize=14, weight='bold', fontfamily='Times New Roman')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=VISUALIZATION_CONFIG['grid_alpha'], linestyle='-', linewidth=0.5, color='gray')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=11,
              framealpha=VISUALIZATION_CONFIG['legend_alpha'])
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"roc_curves{suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"ROC curves saved to: {out_path}")


def plot_tta_curves_from_csvs(output_dir="outputs", epoch=None):
    os.makedirs(output_dir, exist_ok=True)

    # Configure styles
    plt.rcParams['font.family'] = VISUALIZATION_CONFIG['font_family']
    plt.rcParams['font.size'] = VISUALIZATION_CONFIG['font_size']
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    fig, ax = plt.subplots(figsize=(8, 6), dpi=VISUALIZATION_CONFIG['figure_dpi'])

    suffix = _epoch_suffix(epoch)
    file_map = {
        f"fpr_tta_old{suffix}.csv": "TTA (from start)",
        f"fpr_tta_2s{suffix}.csv": "TTA (recent 2s)",
        f"fpr_tta{suffix}.csv": "TTA (abnormal→accident)",
    }

    any_plotted = False
    for fname, label in file_map.items():
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            continue
        data = np.genfromtxt(path, delimiter=',', names=True)
        fpr = data['fpr']
        tta = data['tta']
        ax.plot(fpr, tta, linewidth=VISUALIZATION_CONFIG['line_width'], label=label)
        any_plotted = True

    if not any_plotted:
        return

    ax.set_xlabel('False Positive Rate', fontsize=14, weight='bold', fontfamily='Times New Roman')
    ax.set_ylabel('Time To Accident (s)', fontsize=14, weight='bold', fontfamily='Times New Roman')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=VISUALIZATION_CONFIG['grid_alpha'], linestyle='-', linewidth=0.5, color='gray')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11,
              framealpha=VISUALIZATION_CONFIG['legend_alpha'])
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"fpr_tta_curves{suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"FPR–TTA curves saved to: {out_path}")



def plot_mean_accident_aligned_curve_from_results(
    results,
    output_dir="visualizations",
    epoch=None,
    max_pre_frames=100,
    include_accident_frame=True,
    only_non_test=True,
    auto_max_pre_frames=False,
    normalize_per_video=False,
    upsample_to_frames=False,
    upsample_method="linear",
):
    """
    对验证集的正样本视频，以事故帧对齐，统计事故前第 t 帧的平均预测分数，并保存曲线与CSV。

    参数说明：
    - results: metrics.process 收集的列表，每项包含 keys：'pred', 'target', 'accident_ind', 'is_val', 'is_test' 等。
    - output_dir: 输出目录，默认为 visualizations。
    - epoch: 可选 epoch 后缀。
    - max_pre_frames: 向前统计的最大帧数（窗口大小）。
    - include_accident_frame: 是否包含事故帧作为 t=0。
    - only_non_test: 是否过滤掉测试集样本。
    - auto_max_pre_frames: 是否自动根据最短可用长度裁剪窗口上限。
    - normalize_per_video: 是否对每视频的分数做 [0,1] 归一化后再平均。
    - upsample_to_frames, upsample_method: 预留参数，当前不使用（保持与接口向后兼容）。
    """
    if results is None or len(results) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    # 仅保留验证集正样本，并可选排除测试集
    selected = []
    for r in results:
        if not r.get("target", False):
            continue
        if r.get("is_val", False) is not True:
            continue
        if only_non_test and r.get("is_test", False):
            continue
        pred = r.get("pred", None)
        if pred is None:
            continue
        pred = np.asarray(pred)
        if pred.ndim == 2:
            pred_1d = np.max(pred, axis=-1)
        else:
            pred_1d = pred
        acc_ind = int(r.get("accident_ind", 0))
        if acc_ind < 0 or acc_ind >= len(pred_1d):
            continue
        selected.append((pred_1d, acc_ind))

    if len(selected) == 0:
        return

    # 计算可用的窗口大小
    max_depth_from_data = max(acc_ind + (1 if include_accident_frame else 0) for _, acc_ind in selected)
    T = min(max_pre_frames, max_depth_from_data) if auto_max_pre_frames else max_pre_frames
    T = max(1, int(T))

    # values_by_t[t] 聚合事故前第 t 帧的分数，t=0 对应事故帧
    values_by_t = [[] for _ in range(T)]
    for pred_1d, acc_ind in selected:
        if normalize_per_video:
            pmin = float(np.min(pred_1d))
            pmax = float(np.max(pred_1d))
            denom = (pmax - pmin) if (pmax - pmin) > 1e-12 else 1.0
            pred_use = (pred_1d - pmin) / denom
        else:
            pred_use = pred_1d

        for t in range(T):
            idx = acc_ind - t - (0 if include_accident_frame else 1)
            if idx < 0:
                break
            if idx >= len(pred_use):
                continue
            values_by_t[t].append(float(pred_use[idx]))

    means = [float(np.mean(v)) if len(v) > 0 else np.nan for v in values_by_t]
    counts = [int(len(v)) for v in values_by_t]

    # 生成相对帧坐标：[-T+1, ..., 0] 或 [-T, ..., -1]
    if include_accident_frame:
        rel_frames = list(range(-(T - 1), 0 + 1))
    else:
        rel_frames = list(range(-T, 0))

    # 由于 values_by_t 是按 t=0,1,2...聚合，需翻转以与递增 x 匹配
    rel_frames = np.asarray(rel_frames, dtype=int)
    means = np.asarray(list(reversed(means)), dtype=float)
    counts = np.asarray(list(reversed(counts)), dtype=int)

    # 保存 CSV
    suffix = _epoch_suffix(epoch)
    csv_path = os.path.join(output_dir, f"mean_risk_accident_aligned{suffix}.csv")
    df = pd.DataFrame({
        "relative_frame": rel_frames,
        "mean_score": means,
        "count": counts,
    })
    df.to_csv(csv_path, index=False)

    # 绘制曲线
    plt.rcParams['font.family'] = VISUALIZATION_CONFIG['font_family']
    plt.rcParams['font.size'] = VISUALIZATION_CONFIG['font_size']
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    fig, ax = plt.subplots(figsize=(10, 6), dpi=VISUALIZATION_CONFIG['figure_dpi'])
    ax.plot(rel_frames, means, color=VISUALIZATION_CONFIG['colors']['prediction_line'],
            linewidth=VISUALIZATION_CONFIG['line_width'], label='Mean Prediction (val positives)')
    ax.axvline(x=0, color=VISUALIZATION_CONFIG['colors']['accident_line'], linestyle='--',
               linewidth=2.0, alpha=0.6, label='Accident (t=0)')

    ax.set_xlabel('Frames relative to accident (t)', fontsize=14, weight='bold', fontfamily='Times New Roman')
    ax.set_ylabel('Prediction Score (mean)', fontsize=14, weight='bold', fontfamily='Times New Roman')
    ax.grid(True, alpha=VISUALIZATION_CONFIG['grid_alpha'], linestyle='-', linewidth=0.5, color='gray')
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=11,
              framealpha=VISUALIZATION_CONFIG['legend_alpha'])
    plt.tight_layout()

    img_path = os.path.join(output_dir, f"mean_risk_accident_aligned{suffix}.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Mean accident-aligned curve saved to: {img_path}\nCSV saved to: {csv_path}")

if __name__ == "__main__":
    # 示例用法
    # 假设你的 Tensor 为 N x C x T x H x W
    tensor = torch.rand(5, 3, 30, 128, 128)  # 示例 Tensor
    output_dir = "./videos"
    visualize_tensor_as_videos(tensor, output_dir, fps=10)
    


