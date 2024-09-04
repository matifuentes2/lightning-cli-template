import torch.nn as nn
import torch.nn.functional as F

class ExampleModel(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers):
      super(ExampleModel, self).__init__()
      layers = []
      layers.append(nn.Linear(input_size, hidden_size))
      layers.append(nn.ReLU())
      for _ in range(num_layers - 1):
          layers.append(nn.Linear(hidden_size, hidden_size))
          layers.append(nn.ReLU())
      layers.append(nn.Linear(hidden_size, output_size))
      self.network = nn.Sequential(*layers)

  def forward(self, x):
      x = self.network(x)
      return x
