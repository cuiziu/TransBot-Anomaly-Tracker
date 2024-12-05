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
    :param model_path: 저장된 모델 경로
    :param input_size: 모델 입력 크기
    :param hidden_size: LSTM의 히든 크기
    :param num_classes: 출력 클래스 수
    :return: LSTM 모델
    """
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate_model(model, X_test, y_test, batch_size=8, device='cuda'):
    """
    모델 평가.
    :param model: LSTM 모델
    :param X_test: 테스트 데이터
    :param y_test: 테스트 레이블
    :param batch_size: 배치 크기
    :param device: 실행 디바이스
    :return: 다양한 지표 및 시각화
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Positive 클래스 확률
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(y_batch.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # 지표 계산
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    precision = precision_score(all_labels, all_predictions, zero_division=1)
    recall = recall_score(all_labels, all_predictions, zero_division=1)
    f1 = f1_score(all_labels, all_predictions, zero_division=1)
    auc_roc = roc_auc_score(all_labels, all_probabilities)

    # PRC 곡선
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probabilities)
    prc_auc = np.trapz(recall_vals, precision_vals)

    # 혼동 행렬
    cm = confusion_matrix(all_labels, all_predictions)

    # 결과 출력
    print("Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PRC: {prc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=["Normal", "Assault"]))

    # 그래프 생성
    print("Generating visualizations...")

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"AUC-ROC = {auc_roc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig("roc_curve.png")
    plt.show()

    # PRC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall_vals, precision_vals, label=f"AUC-PRC = {prc_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.savefig("prc_curve.png")
    plt.show()

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Assault"], yticklabels=["Normal", "Assault"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    # YOLOv7 특징 추출기 초기화
    yolo_extractor = YOLOv7FeatureExtractor(
        model_path='yolov7/yolov7-tiny.pt',
        conf_thres=0.65,
        iou_thres=0.6
    )

    # 데이터 로드
    data_loader = DataLoaderModule(data_path='./data/test/', feature_extractor=yolo_extractor)
    _, (X_test, y_test) = data_loader.load_and_split_dataset()

    # 학습된 모델 로드
    model_path = "lstm_model.pth"  # 저장된 모델 경로
    input_size = X_test.shape[-1]
    model = load_trained_model(model_path=model_path, input_size=input_size, hidden_size=128, num_classes=2)

    # 모델 평가
    evaluate_model(model, X_test, y_test, batch_size=8, device='cuda')
