from ultralytics import YOLO
import cv2
import os
from typing import List, Tuple
import json

def save_box_file(res: List[dict], output_dir: str, filename: str) -> str:
    """
    保存预测框信息

    Args:
        res (List[dict]): 预测结果
          - frame_id: 帧编号
          - label: 类别名称
          - left_top: 左上角坐标
          - right_bottom: 右下角坐标
          - conf: 置信度

        output_dir (str): 保存预测框信息的路径
        filename (str): 保存预测框信息的文件名

    Returns:
        output_file (str): 保存预测框信息的路径
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{filename}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
    return output_file


def detect_video(
    model_pt_path: str, video_path: str, output_dir: str
) -> Tuple[str, str]:
    """
    检测视频并保存（带预测框）

    Args:
        model_pt_path (str): 模型权重存放路径
        video_path (str): 需要检测的视频路径
        output_dir (str): 保存检测视频的目录路径

    Returns:
        - output_video_path (str): 保存检测视频的路径
        - output_json_path (str): 保存预测框信息的路径
    """
    if not os.path.exists(model_pt_path):
        raise FileNotFoundError(f"模型权重文件不存在：{model_pt_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    # 构建保存视频路径
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[
        0
    ]  # 获取视频文件的前缀名
    output_video_path = os.path.join(
        output_dir, f"{base_name}_detected.mp4"
    )  # 保存检测后视频的地址
    output_json_filename = base_name + "_detected"  # 保存预测框信息的文件名地址

    # 创建一个 VideoCapture 对象，用于从视频文件中读取帧
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧的维度
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video_path, fourcc, 20.0, (frame_width, frame_height)
    )  # 保存检测后视频的地址

    # 加载模型
    model = YOLO(model_pt_path)

    # 预测
    frame_id = 0
    res = []
    count = 0
    while cap.isOpened():
        status, frame = cap.read()  # 使用 cap.read() 从视频中读取每一帧
        if not status:
            print(f"不存在别的帧")
            print(f"count = {count}, frame_id = {frame_id}")
            break
        result = model.predict(source=frame, conf=0.5)
        # 获取预测框的坐标
        for box in result[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()  # 置信度
            cls_id = int(box.cls[0].item())  # 类别 id
            label = result[0].names[cls_id]  # 类别名称
            res.append(
                {
                    "frame_id": frame_id,
                    "label": label,
                    "left_top": [round(x1, 3), round(y1, 3)],
                    "right_bottom": [round(x2, 3), round(y2, 3)],
                    "conf": round(conf, 3),
                }
            )
        frame_id += 1
        count += 1
        anno_frame = result[0].plot()
        # cv2.imshow("true", anno_frame)
        out.write(anno_frame)  # 写入保存
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    res_json_file = save_box_file(
        res, output_dir=output_dir, filename=output_json_filename
    )
    return output_video_path, res_json_file


if __name__ == "__main__":
    pth_path = "ultralytics/runs/detect/train2/weights/best.pt"  # 训练好的权重文件
    video_path = "inputs/case1/case1.mp4"  # 测试文件
    output_dir = "outputs/case1/detect"  # 保存路径
    output_video_path, output_json_path = detect_video(
        pth_path, video_path, output_dir
    )
    print(
        f"✅ 检测视频保存在：{output_video_path}, 预测框信息保存在：{output_json_path}"
    )