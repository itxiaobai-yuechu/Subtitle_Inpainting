from ultralytics import YOLO
import cv2
import os
from typing import List
import numpy as np


def generate_mask_sequence(
    model_pt_path: str,
    video_path: str,
    output_dir: str,
    target_classes: List[int] = None,
    conf: float = 0.5,
) -> str:
    """
    从视频生成掩码序列：
    - 从第一帧开始读取
    - 找到第一个包含目标类别检测框的帧
    - 从该帧起直到最后一帧，每帧都必须有掩码
    - 若某帧没有检测框，则拷贝前一帧的掩码

    Args:
        model_pt_path (str): YOLO 模型权重路径
        video_path (str): 输入视频路径
        output_dir (str): 掩码保存目录
        target_classes (List[int], optional): 目标类别ID；None 表示所有类别
        conf (float): 置信度阈值

    Returns:
        str: 保存掩码的目录路径
    """
    if not os.path.exists(model_pt_path):
        raise FileNotFoundError(f"模型权重文件不存在：{model_pt_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_pt_path)
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    first_detected = False
    prev_mask = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        results = model.predict(source=frame, conf=conf, verbose=False)
        boxes = results[0].boxes

        has_box = False
        for box in boxes:
            cls_id = int(box.cls[0])
            if (target_classes is None) or (cls_id in target_classes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # 白色填充区域
                has_box = True

        if not first_detected:
            if has_box:
                first_detected = True
                prev_mask = mask.copy()
                frame_path = os.path.join(output_dir, f"{frame_id:05d}.png")
                cv2.imwrite(frame_path, mask)
        else:
            if has_box:
                prev_mask = mask.copy()
            else:
                mask = prev_mask.copy()
            frame_path = os.path.join(output_dir, f"{frame_id:05d}.png")
            cv2.imwrite(frame_path, mask)

        frame_id += 1

    cap.release()
    return output_dir


if __name__ == "__main__":
    pth_path = "/data/aigc/qws/RemoveSubtitles/ultralytics/runs/detect/train2/weights/best.pt"  # 训练好的权重文件
    video_path = "/data/aigc/qws/RemoveSubtitles/case_videos/02.mp4"  # 测试文件
    output_dir = "/data/aigc/qws/RemoveSubtitles/case_videos/02_masks"  # 保存路径
    target_cls = [0]  # 需要处理的类别ID列表
    out_dir = generate_mask_sequence(pth_path, video_path, output_dir, target_cls)
    print(f"✅ 遮罩序列保存至：{out_dir}")