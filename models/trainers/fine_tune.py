import torch
from torch.utils.data import DataLoader, TensorDataset
from load_data import DataLoaderModule
from lstm import LSTMModel
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from fd_yolo import YOLOv7FeatureExtractor

class FineTuner:
    def __init__(self, model_path, input_size, hidden_size, num_classes, device='cuda', learning_rate=0.001):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path))  # 기존 모델 로드
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_history = []  # 손실 기록 추가

    def fine_tune(self, X_train, y_train, epochs=5, batch_size=8):
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.loss_history.append(avg_loss)  # 에포크별 평균 손실 기록
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def evaluate(self, X_test, y_test, batch_size=8):
        dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=batch_size)

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

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print(f"Fine-tuned model saved to {save_path}")


if __name__ == "__main__":
    try:
        # 기존 모델 경로
        existing_model_path = "lstm_model.pth"

        # 추가 데이터 로드
        print("Loading additional data...")
        yolo_extractor = YOLOv7FeatureExtractor(
            model_path='yolov7/yolov7-tiny.pt',
            conf_thres=0.65,
            iou_thres=0.6
        )
        data_loader = DataLoaderModule(data_path='./data/additional/', feature_extractor=yolo_extractor)
        (X_train, y_train), (X_test, y_test) = data_loader.load_and_split_dataset()

        print(f"Additional training data: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
        print(f"Additional testing data: X_test shape={X_test.shape}, y_test shape={y_test.shape}")

        # Fine-tuning
        print("Fine-tuning the model with additional data...")
        fine_tuner = FineTuner(
            model_path=existing_model_path,
            input_size=X_train.shape[-1],
            hidden_size=128,
            num_classes=2
        )
        fine_tuner.fine_tune(X_train=X_train, y_train=y_train, epochs=5)

        # 손실 그래프 그리기
        print("Plotting training loss graph...")
        plt.figure(figsize=(10, 6))
        plt.plot(fine_tuner.loss_history, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Fine-tuning Loss Over Epochs")
        plt.legend()
        plt.grid()
        plt.savefig("fine_tuning_loss.png")
        plt.show()

        # 테스트 데이터 평가
        print("Evaluating model on test data...")
        accuracy = fine_tuner.evaluate(X_test=X_test, y_test=y_test)
        print(f"Test Accuracy on additional data: {accuracy:.4f}")

        # 모델 저장
        fine_tuner.save_model("lstm_finetuned_model.pth")

    except Exception as e:
        print(f"An error occurred: {e}")
