import torch.nn as nn
import torch.optim as optim

class TrainingConfig:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()  # Loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)  # Optimizer

    def compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets)
