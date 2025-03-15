import os
import cv2
import numpy as np
import torch  # 假设 Tensor 是 PyTorch 的格式


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


def visualize_pred_score(result, output_dir, epoch=None, fps=10):
    pred = result["pred"]
    label = result["label"]
    frame_dir = result["frame_dir"]
    filename_tmpl = result["filename_tmpl"]
    frame_inds = result["frame_inds"]
    abnormal_start_frame = result["abnormal_start_frame"]
    abnormal_end_frame = result["abnormal_end_frame"]
    accident_frame = result["accident_frame"]
    video_id = result["video_id"]
    type = result["type"]
    threshold = result["threshold"]

    os.makedirs(output_dir, exist_ok=True)
    # 创建视频保存路径
    if epoch is not None:
        video_path = os.path.join(output_dir, f"{type}_{video_id}_{epoch}.mp4")
    else:
        video_path = os.path.join(output_dir, f"{type}_{video_id}.mp4")

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 MP4 编码
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (640, 360))  # 使用低分辨率确保传输效率

    for i, frame_ind in enumerate(frame_inds):
        frame_path = os.path.join(frame_dir, filename_tmpl.format(frame_ind))
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (640, 360))
        if frame_ind < abnormal_start_frame:
            # 绘制绿色条带
            cv2.rectangle(frame, (0, 0), (640, 36), (0, 255, 0), -1)
            cv2.putText(
                frame,
                "Safe Scene",
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
        elif accident_frame <= frame_ind < abnormal_end_frame:
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
        elif abnormal_end_frame <= frame_ind:
            # 绘制灰色条带
            cv2.rectangle(frame, (0, 0), (640, 36), (128, 128, 128), -1)
            cv2.putText(
                frame,
                "Accident Ended",
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
                l = "?" if i + j >= len(label) else int(label[i + j])
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


if __name__ == "__main__":
    # 示例用法
    # 假设你的 Tensor 为 N x C x T x H x W
    tensor = torch.rand(5, 3, 30, 128, 128)  # 示例 Tensor
    output_dir = "./videos"
    visualize_tensor_as_videos(tensor, output_dir, fps=10)
