import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from fd_yolo import YOLOv7FeatureExtractor
from load_data import DataLoaderModule
from lstm import LSTMModel
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, model, device='cuda', learning_rate=0.001):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)  # 학습률 스케줄러
        self.loss_history = []  # 손실 기록
        self.accuracy_history = []  # 정확도 기록
        self.learning_rate_history = []  # 학습률 기록

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=8):
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # 학습률 저장
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.learning_rate_history.append(current_lr)

            # 학습 손실 기록
            avg_loss = total_loss / len(train_dataloader)
            self.loss_history.append(avg_loss)

            # 검증 정확도 기록
            accuracy = self.evaluate_accuracy(val_dataloader)
            self.accuracy_history.append(accuracy)

            # 스케줄러 업데이트
            self.scheduler.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Learning Rate: {current_lr:.6f}")

    def evaluate_accuracy(self, dataloader):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

        return correct / total


if __name__ == "__main__":
    try:
        # YOLOv7 특징 추출 초기화
        print("Initializing YOLOv7 feature extractor...")
        yolo_extractor = YOLOv7FeatureExtractor(
            model_path='yolov7/yolov7-tiny.pt',
            conf_thres=0.65,
            iou_thres=0.6
        )

        # 데이터 로드 및 분할
        print("Loading and splitting dataset...")
        data_loader = DataLoaderModule(data_path='./data/processed/', feature_extractor=yolo_extractor)
        (X_train, y_train), (X_test, y_test) = data_loader.load_and_split_dataset()

        # 학습 데이터와 검증 데이터 분할
        val_split = 0.2
        val_size = int(len(X_train) * val_split)
        X_val, y_val = X_train[:val_size], y_train[:val_size]
        X_train, y_train = X_train[val_size:], y_train[val_size:]

        print(f"Training set: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
        print(f"Validation set: X_val shape={X_val.shape}, y_val shape={y_val.shape}")
        print(f"Testing set: X_test shape={X_test.shape}, y_test shape={y_test.shape}")

        # 모델 정의
        print("Defining LSTM model...")
        input_size = X_train.shape[-1]
        model = LSTMModel(input_size=input_size, hidden_size=128, num_classes=2)

        # 모델 학습
        print("Starting training...")
        trainer = ModelTrainer(model=model, device='cuda')
        trainer.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, epochs=20)

        # 손실, 정확도, 학습률 그래프 그리기
        print("Plotting combined graph for loss, accuracy, and learning rate...")
        plt.figure(figsize=(12, 6))
        plt.plot(trainer.loss_history, label="Training Loss")
        plt.plot(trainer.accuracy_history, label="Validation Accuracy")
        plt.plot(trainer.learning_rate_history, label="Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Loss, Validation Accuracy, and Learning Rate")
        plt.legend()
        plt.grid()
        plt.savefig("loss_accuracy_lr_graph.png")
        plt.show()

        # 모델 저장
        print("Saving trained model...")
        model_save_path = "lstm_model.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # 테스트 데이터 평가
        print("Evaluating model on test data...")
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        test_dataloader = DataLoader(test_dataset, batch_size=8)
        test_accuracy = trainer.evaluate_accuracy(test_dataloader)
        print(f"Test Accuracy: {test_accuracy:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")
