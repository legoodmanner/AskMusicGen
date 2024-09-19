import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=F.relu, dropout=0.0):
        super(MLP, self).__init__()
        layers = []
        current_size = input_size
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
