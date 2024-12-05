import torch
import numpy as np
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import letterbox
import cv2
from yolov7.utils.general import non_max_suppression

class YOLOv7FeatureExtractor:
    def __init__(self, model_path, device='cuda', conf_thres=0.25, iou_thres=0.45):
        """
        YOLOv7 특징 추출 클래스
        :param model_path: YOLOv7 모델 경로
        :param device: 사용할 디바이스 ('cuda' 또는 'cpu')
        :param conf_thres: 신뢰도 임계값
        :param iou_thres: IoU 임계값
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = attempt_load(model_path, map_location=self.device)
        self.model.eval()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def extract_features_from_video(self, video_path, sequence_length=30):
        """
        비디오에서 YOLOv7 특징 맵을 추출하고 시퀀스 생성.
        :param video_path: 비디오 파일 경로
        :param sequence_length: 시퀀스 길이
        :return: 패딩 처리된 특징 맵 시퀀스
        """
        cap = cv2.VideoCapture(video_path)
        features = []
        feature_dim = 6  # YOLO 감지 결과의 특징 차원 (x, y, w, h, confidence, class)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 전처리
            img = letterbox(frame, new_shape=640)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device).float() / 255.0
            img = img.unsqueeze(0)  # 배치 차원 추가

            # YOLO 감지 수행
            with torch.no_grad():
                pred = self.model(img)[0]
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

            # 필터링 후 특징 맵 저장
            for det in pred:
                if det is not None and len(det):
                    for item in det:
                        # 필요한 정보만 추출 (x, y, w, h, confidence, class)
                        frame_features = item[:feature_dim].cpu().numpy()
                        features.append(frame_features)

        cap.release()

        # 시퀀스 생성 및 패딩
        if len(features) < sequence_length:
            # 시퀀스가 부족한 경우 패딩 처리
            return self.preprocess_sequences([features], max_len=sequence_length, feature_dim=feature_dim)

        sequences = [
            features[i:i + sequence_length] for i in range(len(features) - sequence_length + 1)
        ]

        # 시퀀스 데이터 패딩 처리
        return self.preprocess_sequences(sequences, max_len=sequence_length, feature_dim=feature_dim)

    @staticmethod
    def preprocess_sequences(sequences, max_len=30, feature_dim=6, padding_value=0):
        """
        시퀀스 데이터를 패딩하여 동일한 길이로 만듦.
        :param sequences: 입력 시퀀스 (list of arrays)
        :param max_len: 최대 시퀀스 길이
        :param feature_dim: 특징 차원
        :param padding_value: 패딩 값
        :return: 패딩 처리된 NumPy 배열
        """
        padded_sequences = []
        for seq in sequences:
            padded_seq = np.full((max_len, feature_dim), padding_value, dtype=np.float32)
            for idx, item in enumerate(seq):
                if idx >= max_len:
                    break
                padded_seq[idx] = item
            padded_sequences.append(padded_seq)

        return np.array(padded_sequences, dtype=np.float32)

    def process_data(self, sequences, max_len=30):
        """
        시퀀스를 패딩 처리.
        :param sequences: 입력 시퀀스
        :param max_len: 최대 시퀀스 길이
        :return: 패딩 처리된 NumPy 배열
        """
        return self.preprocess_sequences(sequences, max_len=max_len)
