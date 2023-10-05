import random
import csv
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, data_dir: Path, is_train: bool = True, size: int = 224):
        """
        Args:
            data_dir (Path): データセットのディレクトリ
                訓練 例 Path("data/traffic_sign/traffic_Data/DATA")
                テスト 例 Path("data/traffic_sign/traffic_Data/TEST")
            is_train (bool, optional): フラグ。 ラベルを得る時に使う。
            size (int, optional): 画像のサイズ。デフォルトは224。
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.size = size
        # self.unknown_class = ["40", "41", "42", "45", "49", "52", "56", "57"]
        self.unknown_class = []
        self.img_path_list = []
        self.num_classes = 58 - len(self.unknown_class)
        self.int2label = self.get_int2label()

        # 訓練の場合は data_dir/クラス名/ファイル名.png の形式で画像が保存されている
        # unknown_class は除外して、画像のパスを取得する
        if self.is_train:
            for child in self.data_dir.iterdir():
                if (
                    child.is_dir() and child.name not in self.unknown_class
                ):  # unknown_class は除外
                    temp_path_list = [
                        p for p in child.iterdir() if p.suffix == ".png"
                    ]
                    self.img_path_list.extend(temp_path_list)
        # テストの場合は data_dir/ファイル名.png の形式で画像が保存されている
        else:
            for child in self.data_dir.iterdir():
                # unknown_class は除外
                if (
                    child.suffix == ".png"
                    and str(int(child.name[:3])) not in self.unknown_class
                ):
                    self.img_path_list.append(child)

        assert len(self.img_path_list) > 0, "画像が読み込めませんでした。パスを確認してください。"

        # ラベルの数を取得する。

    def custom_transform(self, img):
        """画像の前処理"""
        img = img.resize(
            (self.size, self.size)
        )  # 参照: https://www.kaggle.com/code/boulahchichenadir/cnn-classification
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )(img)
        return img

    def get_int2label(self):
        int2label = {}
        with open(Path("data/traffic_sign/labels.csv")) as f:
            csv_reader = csv.reader(f)
            next(csv_reader)

            for row in csv_reader:
                key = int(row[0])
                value = row[1]
                int2label[key] = value

        return int2label

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path)
        if self.is_train:
            label = int(img_path.parent.name)

        else:
            # ラベル_ファイル名.png 3桁詰めなので整数に直す
            label = int(img_path.name.split("_")[0])
            label = int(label)

        # 最後に処理
        img = self.custom_transform(img)

        return img, label


# 使用例
def main():
    test_dataset = CustomDataset(
        Path("data/traffic_sign/traffic_Data/TEST"), is_train=False
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)
    inputs, labels = next(iter(test_loader))
    for label in labels.tolist():
        print(test_dataset.int2label[label])
    plt.imshow(inputs[0].permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    main()
