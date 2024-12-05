import numpy as np
import os
from sklearn.model_selection import train_test_split
from fd_yolo import YOLOv7FeatureExtractor

class DataLoaderModule:
    def __init__(self, data_path, feature_extractor, sequence_length=30):
        """
        데이터 로드 및 전처리 클래스
        :param data_path: 데이터 디렉터리 경로
        :param feature_extractor: YOLOv7FeatureExtractor 객체
        :param sequence_length: 시퀀스 길이
        """
        self.data_path = data_path
        self.feature_extractor = feature_extractor
        self.sequence_length = sequence_length

    def load_dataset(self):
        """
        데이터셋을 로드하고 특징 맵 시퀀스와 레이블 생성.
        :return: 특징 맵 시퀀스 (X)와 레이블 (y)
        """
        video_files = []

        # Assult 폴더 (레이블 1)
        assult_path = os.path.join(self.data_path, 'assult')
        for video_file in os.listdir(assult_path):
            video_path = os.path.join(assult_path, video_file)
            if video_file.endswith('.mp4'):
                video_files.append((video_path, 1))  # 레이블 1 추가

        # Normal 폴더 (레이블 0)
        normal_path = os.path.join(self.data_path, 'normal')
        for video_file in os.listdir(normal_path):
            video_path = os.path.join(normal_path, video_file)
            if video_file.endswith('.mp4'):
                video_files.append((video_path, 0))  # 레이블 0 추가

        X, y = [], []

        for video_path, label in video_files:
            try:
                # 비디오에서 특징 추출 (YOLOv7FeatureExtractor 사용)
                sequences = self.feature_extractor.extract_features_from_video(video_path, self.sequence_length)
                X.extend(sequences)  # 추출된 시퀀스를 X에 추가
                y.extend([label] * len(sequences))  # 레이블 추가
            except Exception as e:
                print(f"Error processing {video_path}: {e}")

        # 데이터가 정상적으로 로드되었는지 확인
        if not X or not y:
            raise ValueError("데이터셋 로드 실패: 비어 있는 데이터셋입니다.")

        # 반환할 데이터의 형태를 명시적으로 변환
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def load_and_split_dataset(self, test_size=0.2, random_state=42):
        """
        데이터셋을 로드하고 학습/테스트 데이터로 분할.
        :param test_size: 테스트 데이터 비율 (0~1 사이)
        :param random_state: 랜덤 시드
        :return: 학습 데이터 (X_train, y_train), 테스트 데이터 (X_test, y_test)
        """
        # 전체 데이터 로드
        X, y = self.load_dataset()

        # 학습 데이터와 테스트 데이터로 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return (X_train, y_train), (X_test, y_test)
