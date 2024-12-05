import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from lstm import LSTMModel
from fd_yolo import YOLOv7FeatureExtractor
from load_data import DataLoaderModule


def analyze_video_with_model(model, video_path, output_path, feature_extractor, sequence_length=30, device='cuda'):
    """
    모델의 예측 결과를 비디오로 시각화.
    :param model: 학습된 LSTM 모델
    :param video_path: 입력 비디오 파일 경로
    :param output_path: 저장할 출력 비디오 파일 경로
    :param feature_extractor: YOLOv7 특징 추출 객체
    :param sequence_length: 시퀀스 길이
    :param device: 실행 디바이스
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # 비디오 저장 초기화
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frames = []
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 특징 추출
        features = feature_extractor.extract_features_from_video(video_path, sequence_length)
        if frame_index < len(features):
            sequence = features[frame_index]

            # 모델 예측
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(sequence_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[0, predicted_class].item()

            # 결과 표시
            label = "Normal" if predicted_class == 0 else "Assault"
            color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
            text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # 프레임 저장
        out.write(frame)
        frames.append(frame)
        frame_index += 1

    cap.release()
    out.release()
    print(f"Output video saved to {output_path}")


if __name__ == "__main__":
    # YOLOv7 특징 추출 초기화
    yolo_extractor = YOLOv7FeatureExtractor(
        model_path='yolov7/yolov7-tiny.pt',
        conf_thres=0.65,
        iou_thres=0.6
    )

    # 학습된 모델 로드
    model_path = "lstm_model.pth"
    input_size = 6  # X_test.shape[-1]
    model = LSTMModel(input_size=input_size, hidden_size=128, num_classes=2)
    model.load_state_dict(torch.load(model_path))

    # 입력 비디오 경로 및 출력 비디오 경로
    input_video = "./data/test/assult/cp_assult_579.mp4"
    output_video = "./data/mtest_assult_579.mp4"

    # 비디오 분석
    analyze_video_with_model(model, input_video, output_video, feature_extractor=yolo_extractor)
