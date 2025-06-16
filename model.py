import torch.nn as nn

class DualHeadLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(DualHeadLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.regressor = nn.Linear(hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        price = self.regressor(last_hidden)
        direction = self.classifier(last_hidden)
        return price, direction
