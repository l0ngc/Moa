import torch
import torch.nn as nn

class lstm(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(lstm, self).__init__()
        hidden_size = 100
        self.Lstm = nn.LSTM(num_features, hidden_size // 2, bidirectional=True, batch_first=True, dropout=0.2)

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        # self.dropout1 = nn.Dropout(0.2)
        # self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        #
        # self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        # self.dropout2 = nn.Dropout(0.4)
        # self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        #
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.4)
        self.dense3 = nn.Linear(hidden_size, num_targets)

        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.GeLU = nn.GELU()
        self.RReLU = nn.RReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.batch_norm1(x)
        inp = x.unsqueeze(-2)

        self.Lstm.flatten_parameters()
        lstm, _ = self.Lstm(inp)

        max_pool, _ = torch.max(lstm, 1)
        avg_pool = torch.mean(lstm, 1)

        pooled_output = self.dropout(max_pool)
        pooled_output = self.RReLU(pooled_output)

        x = self.batch_norm3(pooled_output)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x