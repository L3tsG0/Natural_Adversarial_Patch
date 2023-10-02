import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import sys

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.io import read_image
import os

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # すべての画像ファイルを取得
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f.endswith('.png')]

        # ラベルの一覧
        self.classes = ["TraficLight", "Stop", "Speedlimit", "Crosswalk"]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = read_image(img_name)

        # プレフィックスからクラスラベルを取得
        class_label = str(int(self.image_files[idx].split('_')[0]))
        label = self.classes.index(class_label)
        
        if self.transform:
            image = self.transform(image)

        return image, label

class MyModel(nn.Module):
    def __init__(self, n_filters, sz_filters, sz_filters2, sz_pool, n_nodes, n_classes):
        super(MyModel, self).__init__()

        self.n_filters = n_filters

        self.conv1 = nn.Conv2d(3, n_filters, kernel_size=sz_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=sz_filters)
        self.pool = nn.MaxPool2d(kernel_size=sz_pool)
        self.conv3 = nn.Conv2d(n_filters, n_filters // 2, kernel_size=sz_filters2)
        self.conv4 = nn.Conv2d(n_filters // 2, n_filters // 2, kernel_size=sz_filters2)
        self.dropout = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(4 * 4 * (n_filters // 2), n_nodes)
        self.fc1 = nn.Linear(30 * 21 * 21, n_nodes)
        self.fc2 = nn.Linear(n_nodes, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        # print(x.shape) [1, 30, 21, 21]
        x = x.view(-1, 30 * 21 * 21)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def main():
    # ハイパーパラメータを設定
    n_filters = 60
    sz_filters = (5, 5)
    sz_filters2 = (3, 3)
    sz_pool = (2, 2)
    n_nodes = 500
    n_classes = 57 

    # 画像ファイルを開く
    img_path = 'data/traffic_sign/traffic_Data/DATA/0/000_1_0001.png'
    image = Image.open(img_path)
    image = image.resize((100, 100))
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np)
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(torch.float32)
    model = MyModel(n_filters, sz_filters, sz_filters2, sz_pool, n_nodes, n_classes)
    output = model(image_tensor)


if __name__ == '__main__':
    main() 