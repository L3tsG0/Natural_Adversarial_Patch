from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

from src.traffic_data.dataset import CustomDataset


class CNN(nn.Module):
    ## サイズは[224, 224]の画像を想定
    def __init__(self, n_class):
        super(CNN, self).__init__()
        self.name = "cnn"
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, n_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = F.relu(self.conv6(x))
        x = self.pool6(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class simpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(simpleCNN, self).__init__()
        self.name = "simpleCNN"
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# def main():
#     from src.traffic_data.dataset import CustomDataset
#     train_dataset = CustomDataset(Path("data/traffic_sign/traffic_Data/DATA"))
#     data, label  = train_dataset[0]
#     model = CNN(n_class=58)
#     model(data.unsqueeze(0))

# if __name__ == "__main__":
#     main()
