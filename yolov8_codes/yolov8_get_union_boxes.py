from ultralytics import YOLO
import cv2
import os
from typing import List
import numpy as np


def get_union_boxes_from_video(
    model_pt_path: str,
    video_path: str,
    target_cls: int,
    dist_thresh: float = 0.05,  # 比例阈值，例如 0.05 表示 5% 的视频宽度
) -> List[dict]:
    """
    计算视频中指定类别的检测框极大范围，按中心点距离分组合并

    Args:
        model_pt_path (str): YOLO 模型权重路径
        video_path (str): 输入视频路径
        target_cls (int): 目标类别 ID
        dist_thresh (float): 距离阈值，比例值（相对视频宽度）

    Returns:
        union_boxes (list): 每个元素为 dict:
            {
                "left_top": [x1, y1],
                "right_bottom": [x2, y2],
                "cls": target_cls
            }
    """
    if not os.path.exists(model_pt_path):
        raise FileNotFoundError(f"模型权重文件不存在: {model_pt_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 加载模型
    model = YOLO(model_pt_path)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dist_thresh_px = frame_w * dist_thresh  # 阈值转像素

    groups = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.5, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id != target_cls:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                # 分组
                assigned = False
                for g in groups:
                    gx1 = min(b["left_top"][0] for b in g)
                    gy1 = min(b["left_top"][1] for b in g)
                    gx2 = max(b["right_bottom"][0] for b in g)
                    gy2 = max(b["right_bottom"][1] for b in g)
                    g_center = ((gx1 + gx2) / 2, (gy1 + gy2) / 2)

                    if (
                        np.linalg.norm(np.array([cx, cy]) - np.array(g_center))
                        < dist_thresh_px
                    ):
                        g.append(
                            {
                                "left_top": [x1, y1],
                                "right_bottom": [x2, y2],
                                "center": [cx, cy],
                                "cls": cls_id,
                            }
                        )
                        assigned = True
                        break
                if not assigned:
                    groups.append(
                        [
                            {
                                "left_top": [x1, y1],
                                "right_bottom": [x2, y2],
                                "center": [cx, cy],
                                "cls": cls_id,
                            }
                        ]
                    )

    cap.release()

    # 输出极大框
    union_boxes = []
    for g in groups:
        x1 = min(b["left_top"][0] for b in g)
        y1 = min(b["left_top"][1] for b in g)
        x2 = max(b["right_bottom"][0] for b in g)
        y2 = max(b["right_bottom"][1] for b in g)
        union_boxes.append(
            {
                "left_top": [round(float(x1), 2), round(float(y1), 2)],
                "right_bottom": [round(float(x2), 2), round(float(y2), 2)],
                "cls": target_cls,
            }
        )

    return union_boxes


if __name__ == "__main__":
    pth_path = "ultralytics/runs/detect/train2/weights/best.pt"  # 训练好的权重文件
    video_path = "inputs/case1/case1.mp4"  # 测试文件
    target_cls = 0  # 选定 true 类别， false 类别为 1
    dist_thresh = 0.30  # 距离阈值，比例值（相对视频宽度）
    union_boxes = get_union_boxes_from_video(pth_path, video_path, target_cls, dist_thresh)
    print(f"极大框：{union_boxes}")