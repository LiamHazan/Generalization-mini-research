import torch
import torch.nn as nn
num_classes = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#DNN MODELS

class MLP(nn.Module):
    def __init__(self,img_size, mlp_hidden_dim1):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(img_size*img_size, mlp_hidden_dim1),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim1, num_classes)
        )
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, data_point):
        x, label = data_point[0].to(device), data_point[1].to(device)
        x = x.view(1,-1)
        scores = self.MLP(x)
        loss = self.loss_function(scores, label)
        prediction = torch.argmax(scores)
        return loss, prediction

class CNN(nn.Module):
    def __init__(self, img_size, mlp_hidden_dim1, filters):
        super(CNN, self).__init__()
        kernel = 3
        self.conv = nn.Conv2d(1, filters, kernel_size=kernel)
        self.pooling = nn.MaxPool2d(2)
        self.MLP = nn.Sequential(
            nn.Linear(filters*(15**2), mlp_hidden_dim1),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim1, num_classes)
        )
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, data_point):
        x, label = data_point[0].unsqueeze(0).unsqueeze(0).to(device), data_point[1].to(device)
        x = self.conv(x)
        # print(x.shape)
        x = self.pooling(x)
        # print(x.shapeq)
        x = x.view(1,-1)
        scores = self.MLP(x)
        loss = self.loss_function(scores, label)
        prediction = torch.argmax(scores)
        return loss, prediction

