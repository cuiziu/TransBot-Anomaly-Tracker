import os
import cv2
import xml.etree.ElementTree as ET
import torch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 경로 설정
video_folder = "./data/preprocess_test/video/"
label_folder = "./data/preprocess_test/label/"
output_clips_folder = "./data/pre_test/"

# 출력 폴더 생성
os.makedirs(output_clips_folder, exist_ok=True)

# YOLO 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 유틸리티 함수: 시간 형식을 초로 변환
def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

# XML에서 행동 구간 추출
def extract_action_frames_from_event(xml_path, video_fps):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    action_frames = []

    for event in root.findall(".//event"):
        event_name = event.find('eventname').text
        start_time_str = event.find('starttime').text
        duration_str = event.find('duration').text

        if start_time_str and duration_str:
            start_time = time_to_seconds(start_time_str)
            duration = time_to_seconds(duration_str)
            start_frame = int(start_time * video_fps)
            end_frame = int((start_time + duration) * video_fps)
            action_frames.append((event_name, start_frame, end_frame))

    return action_frames

# 구간 클립 생성 및 저장
def save_action_clips(video_path, action_frames):
    temp_clips = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for i, (event_name, start_frame, end_frame) in enumerate(action_frames):
        output_path = f"{video_path}_clip_{i + 1}.mp4"
        temp_clips.append(output_path)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()

    cap.release()
    return temp_clips

# ROI 기반 클립 저장
def save_roi_clip(video_path, output_path, confidence_threshold=0.7):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    min_x, min_y, max_x, max_y = width, height, 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results.pred[0]:
            confidence = result[4]
            if int(result[5]) == list(model.names.values()).index('person') and confidence >= confidence_threshold:
                x1, y1, x2, y2 = map(int, result[:4])
                min_x, min_y = min(min_x, x1), min(min_y, y1)
                max_x, max_y = max(max_x, x2), max(max_y, y2)

    cap.release()

    roi_width = max_x - min_x
    roi_height = max_y - min_y

    if roi_width <= 0 or roi_height <= 0:
        print(f"No valid ROI for {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (roi_width, roi_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        roi_frame = frame[min_y:max_y, min_x:max_x]
        out.write(roi_frame)

    cap.release()
    out.release()
    print(f"Saved ROI clip: {output_path}")

# 메인 처리 루프
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(video_folder, video_file)
        label_file = os.path.join(label_folder, os.path.splitext(video_file)[0] + ".xml")

        if os.path.exists(label_file):
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            action_frames = extract_action_frames_from_event(label_file, fps)
            temp_clips = save_action_clips(video_path, action_frames)

            for temp_clip in temp_clips:
                # 기존 파일 이름에서 새로운 형식으로 변환
                base_name = os.path.basename(temp_clip).split('_clip')[0]
                roi_output_path = os.path.join(output_clips_folder, f"cp_{base_name}.mp4")
                save_roi_clip(temp_clip, roi_output_path)

                # 임시 클립 삭제
                os.remove(temp_clip)