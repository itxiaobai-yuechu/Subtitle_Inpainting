from ultralytics import YOLO
import cv2
import os
from typing import List
import numpy as np

def detect_and_save_edges(
    model_pt_path: str,
    video_path: str,
    output_dir: str,
    target_classes: List[int] = None,
) -> str:
    """
    对视频逐帧检测，仅在指定类别的预测框内进行边缘检测，保存每一帧的黑白图片

    Args:
        model_pt_path (str): YOLO 模型权重路径
        video_path (str): 输入视频路径
        output_dir (str): 保存帧图片的目录
        target_classes (List[int]): 需要处理的类别ID列表，例如 [0,1]；如果为 None 则处理所有类别

    Returns:
        output_dir (str): 保存的图片目录
    """
    if not os.path.exists(model_pt_path):
        raise FileNotFoundError(f"模型权重文件不存在：{model_pt_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    model = YOLO(model_pt_path)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(source=frame, conf=0.5, verbose=False)
        boxes = result[0].boxes

        # 创建全黑图（单通道灰度）
        edge_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        if len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])  # 获取类别ID
                if (target_classes is None) or (cls_id in target_classes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    roi = frame[y1:y2, x1:x2]

                    # 转灰度 + 边缘检测
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 100, 200)

                    # 把边缘结果放回对应位置
                    edge_frame[y1:y2, x1:x2] = edges

        # 保存每帧为 PNG
        # frame_path = os.path.join(output_dir, f"frame_{frame_id:05d}.png")
        frame_path = os.path.join(output_dir, f"{frame_id:05d}.png")
        cv2.imwrite(frame_path, edge_frame)

        frame_id += 1

    cap.release()
    return output_dir

if __name__ == "__main__":
    pth_path = "/data/aigc/qws/RemoveSubtitles/ultralytics/runs/detect/train2/weights/best.pt"  # 训练好的权重文件
    video_path = "/data/aigc/qws/RemoveSubtitles/case_videos/02.mp4"  # 测试文件
    output_dir = "/data/aigc/qws/RemoveSubtitles/case_videos/02_edge"  # 保存路径
    target_cls = [0]  # 需要处理的类别ID列表
    out_dir = detect_and_save_edges(pth_path, video_path, output_dir, target_cls)
    print(f"✅ 边缘检测的结果保存至：{out_dir}")