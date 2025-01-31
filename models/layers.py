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
            layers.append(nn.Tanh() if activation == F.relu else activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
from madmom.features.beats import DBNBeatTrackingProcessor

def dbnProcessor(activations, fps):
    # activation.shape = (B, seq_len, 2)
    proc = DBNBeatTrackingProcessor(fps=fps)
    result = []
    for act in activations:
        act = act.detach().cpu().numpy()
        result += [proc(act)] # pred
    return result #list of beat position in seconds e.g. [1.24, 1.45, 1.78, ...]
