import cv2
import numpy as np


def wrap_text(text, font, font_scale, thickness, max_width):
    """文本自动换行函数"""
    lines = []
    current_line = []
    current_width = 0

    # 按空格分割单词
    words = text.split()

    for word in words:
        # 测试添加当前单词后的宽度
        test_line = ' '.join(current_line + [word])
        (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

        if w <= max_width:
            current_line.append(word)
            current_width = w
        else:
            if current_line:  # 保存当前行
                lines.append(' '.join(current_line))
            current_line = [word]
            current_width = cv2.getTextSize(word, font, font_scale, thickness)[0][0]

    if current_line:  # 添加最后一行
        lines.append(' '.join(current_line))

    return lines


def process_videos(input_paths, output_path, labels, skip_frames=4):
    caps = [cv2.VideoCapture(path) for path in input_paths]

    # 获取第一个有效视频参数
    base_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    base_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caps[0].get(cv2.CAP_PROP_FPS)

    # 初始化输出尺寸参数
    text_config = {
        'font': cv2.FONT_HERSHEY_SIMPLEX,
        'font_scale': 0.7,
        'thickness': 1,
        'color': (255, 255, 255),
        'bg_color': (0, 0, 0),
        'margin': 20  # 左右边距
    }

    # 技术说明文本
    caption_text = "DRIVE and CAP predicted a large number of false alarms before the anomaly appears (green bar)."

    # 预计算文本换行 (在初始化阶段完成)
    output_width = base_width + 200  # 文本栏200像素
    max_text_width = output_width - 2 * text_config['margin']
    wrapped_lines = wrap_text(caption_text,
                              text_config['font'],
                              text_config['font_scale'],
                              text_config['thickness'],
                              max_text_width)

    # 计算caption高度
    (_, line_height), _ = cv2.getTextSize("Test", text_config['font'],
                                          text_config['font_scale'],
                                          text_config['thickness'])
    line_spacing = 10
    caption_height = len(wrapped_lines) * (line_height + line_spacing) + 2 * text_config['margin']

    # 最终输出尺寸
    output_height = base_height * 3 + caption_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    # 跳过前4帧（仅前两个视频）
    for i in range(2):
        for _ in range(skip_frames):
            caps[i].read()

    while True:
        frames = []
        all_finished = True

        # 处理每个视频流
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                # 生成黑帧保持尺寸
                frame = np.zeros((base_height, base_width, 3), dtype=np.uint8)
            else:
                all_finished = False

            # 添加左侧标签
            frame = np.ascontiguousarray(frame)
            text_panel = np.zeros((frame.shape[0], 200, 3), dtype=np.uint8)
            cv2.putText(text_panel, labels[i],
                        (10, frame.shape[0] // 2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            frames.append(cv2.hconcat([text_panel, frame]))

        if all_finished:
            break

        # 垂直拼接视频
        video_stack = cv2.vconcat(frames)

        # 创建caption面板
        caption_panel = np.zeros((caption_height, output_width, 3), dtype=np.uint8)
        caption_panel[:] = text_config['bg_color']

        # 计算文本起始位置
        total_text_height = len(wrapped_lines) * (line_height + line_spacing)
        start_y = (caption_height - total_text_height) // 2 + line_height

        # 绘制多行文本
        for idx, line in enumerate(wrapped_lines):
            (tw, th), _ = cv2.getTextSize(line, text_config['font'],
                                          text_config['font_scale'],
                                          text_config['thickness'])
            x = (output_width - tw) // 2
            y = start_y + idx * (th + line_spacing)
            cv2.putText(caption_panel, line, (x, y),
                        text_config['font'],
                        text_config['font_scale'],
                        text_config['color'],
                        text_config['thickness'],
                        cv2.LINE_AA)

        # 组合最终画面
        final_frame = cv2.vconcat([video_stack, caption_panel])
        out.write(final_frame)

    # 释放资源
    for cap in caps:
        cap.release()
    out.release()


if __name__ == "__main__":
    input_files = ["DRIVE_003483.mp4", "CAP_003483.mp4", "snippnet_003483.mp4"]
    video_labels = ["DRIVE", "CAP", "Ours"]
    process_videos(input_files, "final_output.mp4", video_labels, 4)