import torch
from torchmetrics import F1Score
    
class F1ScoreMeter:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.prediction = []
        self.target = []

    def add(self, prediction, target):
        self.prediction.extend(prediction.cpu().detach().tolist())
        self.target.extend(target.cpu().detach().tolist())

    def value(self, average):
        f1 = F1Score(self.num_classes, average=average)
        return f1(torch.tensor(self.prediction), torch.tensor(self.target))
