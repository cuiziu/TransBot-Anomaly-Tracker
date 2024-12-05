import cv2
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report
)
import seaborn as sns
from lstm import LSTMModel  # 학습에 사용한 모델 클래스
from fd_yolo import YOLOv7FeatureExtractor  # YOLO 특징 추출기
from load_data import DataLoaderModule  # 데이터 로더


def load_trained_model(model_path, input_size, hidden_size, num_classes):
    """
    저장된 모델을 로드.
    """
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate_model_on_video(model, video_path, feature_extractor, frame_skip=2, sequence_length=30, device='cuda'):
    """
    모델을 비디오에 적용하여 분석 결과를 시각화.
    :param model: 학습된 LSTM 모델
    :param video_path: 입력 비디오 경로
    :param feature_extractor: YOLOv7 특징 추출기
    :param frame_skip: 처리할 프레임 간격
    :param sequence_length: LSTM 시퀀스 길이
    :param device: 실행 디바이스
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    frame_index = 0
    features = []
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 간격 처리
        if frame_index % frame_skip == 0:
            # YOLO 특징 추출
            frame_features = feature_extractor.extract_features_from_frame(frame)
            if frame_features is not None:
                features.append(frame_features)

            # 충분한 시퀀스가 준비되었을 때 LSTM 모델에 전달
            if len(features) >= sequence_length:
                sequence = features[-sequence_length:]  # 마지막 시퀀스 가져오기
                sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(sequence_tensor)
                    predicted_class = torch.argmax(output, dim=1).item()
                    predictions.append(predicted_class)

                # 결과를 프레임에 표시
                label = "Normal" if predicted_class == 0 else "Assault"
                color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
                confidence = torch.softmax(output, dim=1).max().item()
                text = f"{label} ({confidence:.2f})"
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        frame_index += 1

    cap.release()
    return predictions


if __name__ == "__main__":
    # YOLOv7 특징 추출기 초기화
    yolo_extractor = YOLOv7FeatureExtractor(
        model_path='yolov7/yolov7-tiny.pt',
        conf_thres=0.65,
        iou_thres=0.6
    )

    # 학습된 모델 로드
    model_path = "lstm_model.pth"  # 저장된 모델 경로
    input_size = 6  # 입력 특징 수 (YOLO 출력 차원)
    model = load_trained_model(model_path=model_path, input_size=input_size, hidden_size=128, num_classes=2)

    # 비디오 경로 설정
    video_path = "./data/test_video.mp4"  # 입력 비디오 파일
    frame_skip = 3  # 처리할 프레임 간격

    # 비디오 분석
    print("Analyzing video...")
    predictions = evaluate_model_on_video(
        model=model,
        video_path=video_path,
        feature_extractor=yolo_extractor,
        frame_skip=frame_skip,
        sequence_length=30,
        device='cuda'
    )

    print("Predictions:", predictions)
