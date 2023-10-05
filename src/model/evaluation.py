from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.cnn import simpleCNN
from src.traffic_data.dataset import CustomDataset


def load_cnn_model(
    path: Path, num_classes: int = 58, device: torch.device = None
) -> simpleCNN:
    """パラメータのみ保存されたモデルを読み込む"""
    model = simpleCNN(num_classes)
    model.load_state_dict(torch.load(map_location=device, f=path))
    return model


def load_model(path: Path, device: torch.device = None) -> torch.nn.Module:
    """モデル構造を含めて保存されたモデルを読み込む"""
    model = torch.load(map_location=device, f=path)
    return model


def compute_cnn_test_accuracy(device: torch.device):
    test_dataset = CustomDataset(
        Path("data/traffic_sign/traffic_Data/TEST"), is_train=False
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

    model = load_model(
        Path("src/model/trained_CNN.pth"),
        device=device,
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs, 1)
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
        print(f"Accuracy: {correct / total}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    compute_cnn_test_accuracy(device)


if __name__ == "__main__":
    main()
