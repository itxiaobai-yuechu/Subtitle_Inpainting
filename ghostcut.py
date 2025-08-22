import requests
import hashlib
import json
import time
from yolov8_codes.yolov8_get_union_boxes import get_union_boxes_from_video
import cv2
from moviepy.editor import VideoFileClip
import os
import mimetypes

def remove_subtitle(appKey, appSecret, urls, videoInpaintMasks, model):
    url="https://api.zhaoli.com/v-w-c/gateway/ve/work/free"
    body=json.dumps({
        "urls": urls,
        "videoInpaintMasks": videoInpaintMasks,
        "extraOptions": {
            "extra_inpaint_config": {
                "model": model
            }
        },
        "resolution": "1080p",
        "needChineseOcclude": 2,
        "videoInpaintLang": "all"
    })
    md5_1 = hashlib.md5()
    md5_1.update(body.encode('utf-8'))
    body_md5hex = md5_1.hexdigest()
    md5_2 = hashlib.md5()
    body_md5hex = (body_md5hex + appSecret).encode('utf-8')
    md5_2.update(body_md5hex)
    sign = md5_2.hexdigest()
    headers = {
        'Content-type': 'application/json',
        'AppKey': appKey,
        'AppSign': sign,
    }
    result=requests.post(url, data=body, headers= headers).json()
    return result

def query_task_status(appKey, appSecret, idWorks):
    url = "https://api.zhaoli.com/v-w-c/gateway/ve/work/status"
    body = json.dumps({
        "idWorks": idWorks
    })
    md5_1 = hashlib.md5()
    md5_1.update(body.encode('utf-8'))
    body_md5hex = md5_1.hexdigest()
    md5_2 = hashlib.md5()
    body_md5hex = (body_md5hex + appSecret).encode('utf-8')
    md5_2.update(body_md5hex)
    sign = md5_2.hexdigest()
    headers = {
        'Content-type': 'application/json',
        'AppKey': appKey,
        'AppSign': sign,
    }

    while True:
        try:
            resp = requests.post(url, data=body, headers=headers, timeout=10).json()
            if resp.get("code") != 1000:
                print("接口返回错误：", resp.get("msg"))
                return None
            data_list = resp.get("body", {}).get("content", [])
            if not data_list:
                print("没有任务数据")
                return None
            task = data_list[0]  # 假设一次只查一个
            process_status = task.get("processStatus", -1)
            if process_status < 1:
                print("任务处理中，15秒后再次查询...")
                time.sleep(15)
                continue
            elif process_status == 1:
                print("任务成功")
                return task.get("videoUrl")
            else:
                print("任务失败，状态码：", process_status)
                return None
        except Exception as e:
            print("请求异常：", e)

def upload_file(API_KEY, SECRET_KEY, File_path):
    url = "https://dev01-ai-orchestration.tec-develop.cn/api/ai/s3/v1/upload"
    headers = {
        "X-API-Key": API_KEY,
        "X-Secret-Key": SECRET_KEY
    }
    filename = os.path.basename(File_path)
    mime_type, _ = mimetypes.guess_type(File_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    with open(File_path, "rb") as f:
        files = {
            "file": (filename, f, mime_type)
        }
        response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        data = response.json()
        print("上传成功，链接为：", data.get("endpoint"))
    else:
        print("上传失败，状态码：", response.status_code)
        print("响应内容：", response.text)
    return data.get("endpoint")

if __name__ == "__main__":
    # step1 将视频上传到华为oss
    API_KEY = "8510521248724897"
    SECRET_KEY = "gip29ldl26cpzuwvt2slmtffldhjh45z"
    video_path = "inputs/case1/case1.mp4"
    urls = [upload_file(API_KEY, SECRET_KEY, video_path)]

    # step2 获取视频中的目标框
    pth_path = "ultralytics/runs/detect/train2/weights/best.pt"  # 训练好的权重文件
    target_cls = 0  # 选定 true 类别， false 类别为 1
    dist_thresh = 0.30  # 距离阈值，比例值（相对视频宽度）
    union_boxes = get_union_boxes_from_video(pth_path, video_path, target_cls, dist_thresh)
    x1_list = [box["left_top"][0] for box in union_boxes]
    y1_list = [box["left_top"][1] for box in union_boxes]
    x2_list = [box["right_bottom"][0] for box in union_boxes]
    y2_list = [box["right_bottom"][1] for box in union_boxes]

    # step3 获取用于ghostcut的mask
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1 = min(x1_list) / width
    y1 = min(y1_list) / height
    x2 = max(x2_list) / width
    y2 = max(y2_list) / height

    # step4 调用ghostcut接口
    clip = VideoFileClip(video_path)
    duration = clip.duration
    appKey="FUxKpXAnvbOmeLjq8PX0G57KB5wLYvsH"
    appSecret=r"7*=%F1HfB7la2Le&_%-SI#WnNA)F@%$-"
    videoInpaintMasks = [
        {
            "type": "remove_only_ocr",
            "start": 0, # 框开始秒数
            "end": duration, # 框结束秒数
            "region": [
                [x1, y1],
                [x2, y1],
                [x1, y2],
                [x2, y2]
            ] # 左上、右上、左下、右下。全屏消除模式不要传
        }
    ] # advanced_lite最多三个框，advanced和advanced_large_box最多一个框，全屏模式不需要框
    model = "advanced" # advanced_lite，advanced，advanced_large_box，advanced_full
    result = remove_subtitle(appKey, appSecret, urls, videoInpaintMasks, model)
    print("请求成功，返回参数为：", result)
    idWorks = [item["id"] for item in result["body"]["dataList"]]
    video_url = query_task_status(appKey, appSecret, idWorks)
    print("处理后的视频链接：", video_url)