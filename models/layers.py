import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=F.relu, dropout=0.0):
        super(MLP, self).__init__()
        layers = []
        current_size = input_size
        # layers.append(nn.BatchNorm1d(current_size, affine=True))
        layers.append(nn.LazyBatchNorm1d(current_size))
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU() if activation == F.relu else activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, bidirectional=True, num_layers=2, dropout=0.0):
        super(LSTM, self).__init__()
        assert isinstance(hidden_sizes, int)
        self.bn = nn.LazyBatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_sizes, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_sizes * (2 if bidirectional else 1), output_size)

        self.tmp = MLP(input_size, hidden_sizes, output_size, activation=F.relu, dropout=dropout)
    def forward(self, x):
        '''
        x: [batch_size, seq_len, input_size]
        '''
        # x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        # lstm_out, _ = self.lstm(x)
        # out = self.fc(lstm_out[:, -1, :])
        lstm_out = x.mean(dim=1)
        out = self.tmp(lstm_out)

        return out

    
from madmom.features.beats import DBNBeatTrackingProcessor

def dbnProcessor(activations, fps):
    # activation.shape = (B, seq_len, 2)
    proc = DBNBeatTrackingProcessor(fps=fps)
    result = []
    for act in activations:
        act = act.detach().cpu().numpy()
        result += [proc(act)] # pred
    return result #list of beat position in seconds e.g. [1.24, 1.45, 1.78, ...]


