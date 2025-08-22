from yolov8_codes.yolov8_get_union_boxes import get_union_boxes_from_video
import cv2
import os
from PIL import Image
import numpy as np
import os
import subprocess

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件 {video_path}")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"{frame_idx:05d}.png"
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
    cap.release()
    print(f"提取完成，共保存 {frame_idx} 帧到 {output_dir}")

def create_mask(W, H, x1_list, y1_list, x2_list, y2_list, frames_count, output_dir):
    # 根据list创造mask
    mask = np.zeros((H, W), dtype=np.uint8)
    for x1, y1, x2, y2 in zip(x1_list, y1_list, x2_list, y2_list):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        x1, x2 = sorted((max(0, x1), min(W - 1, x2)))
        y1, y2 = sorted((max(0, y1), min(H - 1, y2)))
        mask[y1:y2+1, x1:x2+1] = 255
    mask_img = Image.fromarray(mask)

    # 复制frames_count张mask
    os.makedirs(output_dir, exist_ok=True)
    for i in range(frames_count):
        mask_img.save(os.path.join(output_dir, f"{i:05d}.png"))
    print(f"保存 {frames_count} mask 到 {output_dir}")


if __name__ == "__main__":
    # step1 获取视频所有帧
    video_path = "inputs/case1/case1.mp4"  # 测试文件
    output_dir = "outputs/case1"

    extract_frames(video_path, os.path.join(output_dir, "frames"))

    # step2 获取视频中的目标框
    pth_path = "ultralytics/runs/detect/train2/weights/best.pt"  # 训练好的权重文件
    target_cls = 0  # 选定 true 类别， false 类别为 1
    dist_thresh = 0.30  # 距离阈值，比例值（相对视频宽度）
    union_boxes = get_union_boxes_from_video(pth_path, video_path, target_cls, dist_thresh)
    x1_list = [box["left_top"][0] for box in union_boxes]
    y1_list = [box["left_top"][1] for box in union_boxes]
    x2_list = [box["right_bottom"][0] for box in union_boxes]
    y2_list = [box["right_bottom"][1] for box in union_boxes]

    # step3 获取用于inpaint的mask
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    create_mask(width, height, x1_list, y1_list, x2_list, y2_list, frame_count, os.path.join(output_dir, "frames_mask"))

    # step4 inpainting阶段
    os.makedirs(os.path.join(output_dir, "result_video"), exist_ok=True)
    cmd = [
        "python",
        "propainter/inference_propainter.py",
        "--video", video_path,
        "--mask", os.path.join(output_dir, "frames_mask"),
        "--subvideo_length", "30",
        "--save_fps", str(fps),
        "--fp16",
        "--save_root", os.path.join(output_dir, "result_video")
    ] # --fp16会减少显存占用，但是也会影响精度
    subprocess.run(cmd, check=True)