import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        LSTM 모델 정의
        :param input_size: 입력 특징 크기
        :param hidden_size: LSTM 은닉층 크기
        :param num_classes: 출력 클래스 수
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 마지막 시퀀스 출력
        out = self.fc(out)
        return out
