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
    train_loader, test_loader = config.train_loader, config.test_loader
    criterion, optimizer = config.criterion, config.optimizer
    model = config.model.to(config.device)
    writer = SummaryWriter(log_dir=f"log{config.model.name}/")  # 途中経過を確認する

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, total=len(train_loader), leave=True)
        train_loop.set_description(f"Epoch [{epoch}/{config.epochs}]")
        total_train_batch = len(train_loader)

        # 訓練ループ
        for train_batch_idx, (X_batch, y_batch) in enumerate(train_loop):
            X_batch, y_batch = X_batch.to(config.device), y_batch.to(
                config.device
            )
            y_batch_pred = model(X_batch)
            loss = criterion(y_batch_pred, y_batch)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(loss=loss.item())

            if train_batch_idx % (total_train_batch // 100) == 0:
                # TensorBoardにログを書き出す
                writer.add_scalar(
                    "train loss",
                    loss.item(),
                    global_step=epoch * len(train_loader)
                    + train_batch_idx * train_loader.batch_size,
                )

        train_loss /= len(train_loader)
        writer.add_scalar(
            "train loss(per epoch)",
            train_loss,
            global_step=epoch + 1,
        )

        model.eval()
        test_loss = 0.0
        test_loop = tqdm(test_loader, total=len(test_loader), leave=True)

        with torch.no_grad():
            for test_batch_idx, (X_batch, y_batch) in enumerate(test_loop):
                X_batch, y_batch = X_batch.to(config.device), y_batch.to(
                    config.device
                )
                y_batch_pred = model(X_batch)
                loss = criterion(y_batch_pred, y_batch)
                test_loss += loss.item()

                test_loop.set_description(
                    f"Epoch [{epoch}/{config.epochs}] test"
                )
                test_loop.set_postfix(loss=loss.item())

        test_loss /= len(test_loader)

        writer.add_scalar(
            "test loss(per epoch)", test_loss, global_step=epoch + 1
        )

        torch.save(
            model.state_dict(),
            f"{config.model.name}_{epoch}.pth",
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
        device=torch.device('mps'),
    )
    train(train_config)


if __name__ == "__main__":
    main()
