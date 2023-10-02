from pathlib import Path
from dataclasses import dataclass

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.model.cnn import CNN
from src.traffic_data.dataset import CustomDataset


@dataclass
class TrainConfig:
    model: torch.nn.Module
    epochs: int
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    device: str


def train(config: TrainConfig):
    writer = SummaryWriter(log_dir=f"log_{config.model.name}/")  # 途中経過を確認する
    train_loader, test_loader = config.train_loader, config.test_loader
    criterion, optimizer = config.criterion, config.optimizer
    model = config.model.to(config.device)

    for epoch in range(config.epochs):
        train_loop = tqdm(train_loader, total=len(train_loader), leave=True)
        train_loop.set_description(f"Epoch [{epoch}/{config.epochs}]")
        num_train_batch = len(train_loader)

        # 訓練ループ
        train_correct, train_total = 0, 0  # accuracy計算用
        train_loss = 0.0  # epochごとのloss計算用
        for i, data in enumerate(train_loop):
            inputs, labels = data
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            optimizer.zero_grad()

            preds = model(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted_labels = torch.argmax(preds, dim=1)

            train_correct += (predicted_labels == labels).sum().item()
            train_total += labels.size(0)

            train_loop.set_postfix(loss=loss.item())

            if i % (num_train_batch // 100) == 0:
                writer.add_scalar(
                    "train loss",
                    loss.item(),
                    global_step=epoch * len(train_loader)
                    + i * train_loader.batch_size,
                )

        train_loss /= len(train_loader)
        writer.add_scalar(
            "train loss(per epoch)",
            train_loss,
            global_step=epoch + 1,
        )

        with torch.no_grad():
            test_loop = tqdm(test_loader, total=len(test_loader), leave=True)

            # テストループ
            test_correct, test_total = 0, 0
            test_loss = 0.0
            for inputs, labels in test_loop:
                inputs, labels = inputs.to(config.device), labels.to(
                    config.device
                )
                preds = model(inputs)
                loss = criterion(preds, labels)

                test_loss += loss.item()

                predicted_labels = torch.argmax(preds, dim=1)
                test_correct += (predicted_labels == labels).sum().item()
                test_total += labels.size(0)

                test_loop.set_description(
                    f"Epoch [{epoch}/{config.epochs} test"
                )
                test_loop.set_postfix(loss=loss.item())

            test_accuracy = test_correct / test_total
            test_loss /= len(test_loader)

            writer.add_scalar(
                "test loss(per epoch)", test_loss, global_step=epoch + 1
            )
            writer.add_scalar(
                "test accuracy(per epoch)", test_accuracy, global_step=epoch + 1
            )

        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                Path(f"src/model/{config.model.name}_{epoch}.pth"),
            )

    writer.close()


def main():
    """
    args:
        log_dir_root: tensorboardのログを保存するディレクトリこのディレクトリのしたに/model_name/が作成されます
        model: 学習するモデル
        model_name: モデルの名前 tensorboardのログやチェックポイント(mks)の保存に使われます
    """

    train_dataset = CustomDataset(Path("data/traffic_sign/traffic_Data/DATA"))
    test_dataset = CustomDataset(
        Path("data/traffic_sign/traffic_Data/TEST"), is_train=False
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=32, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

    model = CNN(train_dataset.num_classes)

    train_config = TrainConfig(
        model=CNN(train_dataset.num_classes),
        epochs=100,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        criterion=torch.nn.CrossEntropyLoss(),
        train_loader=train_loader,
        test_loader=test_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    train(train_config)


if __name__ == "__main__":
    main()
