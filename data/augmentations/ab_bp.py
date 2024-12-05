import os
import cv2
import xml.etree.ElementTree as ET

# 경로 설정
video_folder = "./data/preprocess_test/video/"
label_folder = "./data/preprocess_test/label/"
output_clips_folder = "./data/pre_test/"

# 출력 폴더 생성
os.makedirs(output_clips_folder, exist_ok=True)


# 유틸리티 함수: 시간 형식을 초로 변환
def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s


# XML에서 행동 시작 프레임 추출
def get_first_action_start_frame(xml_path, video_fps):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        start_frames = []

        for event in root.findall(".//event"):
            start_time_str = event.find('starttime').text
            if start_time_str:
                start_time = time_to_seconds(start_time_str)
                start_frame = int(start_time * video_fps)
                start_frames.append(start_frame)

        if start_frames:
            return min(start_frames)  # 가장 이른 시작 프레임 반환
    except Exception as e:
        print(f"Error processing XML file {xml_path}: {e}")
    return None


# 행동 전 구간 클립 저장
def save_pre_action_clip(video_path, start_frame, output_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_num in range(start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
        print(f"Saved pre-action clip: {output_path}")
    except Exception as e:
        print(f"Error saving pre-action clip for {video_path}: {e}")


# 메인 처리 루프
for video_file in os.listdir(video_folder):
    try:
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_folder, video_file)
            label_file = os.path.join(label_folder, os.path.splitext(video_file)[0] + ".xml")

            if os.path.exists(label_file):
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video file {video_path}")

                fps = int(cap.get(cv2.CAP_PROP_FPS))
                cap.release()

                start_frame = get_first_action_start_frame(label_file, fps)
                if start_frame is not None:
                    output_path = os.path.join(output_clips_folder, f"pre_{os.path.basename(video_file)}")
                    save_pre_action_clip(video_path, start_frame, output_path)
            else:
                print(f"Label file not found for video: {video_file}")
    except Exception as e:
        print(f"Error processing video file {video_file}: {e}")
