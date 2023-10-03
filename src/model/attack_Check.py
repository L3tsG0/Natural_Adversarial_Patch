from torch import tensor
import torch
from pathlib import Path
from src.model.attacker import (
    CutoutEdge,
    custom_transform,
    paste_rotated_image_with_alpha_at_center,
)
from src.model.cnn import simpleCNN
from src.model.evaluation import load_cnn_model
from PIL import Image
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# setting from the result of PEPG
position = torch.tensor([0.0000, 0.5173, 0.0000])

PatchCenter: tuple[Tensor, Tensor] = (position[0], position[1])
patch_angle_ratio: Tensor = position[2]
patch_size_ratio = 0.3


# load batterfly img
batterfly_img_path = Path("src/model/Attack_data/0010001.png")
batterfly_img: Image = CutoutEdge(batterfly_img_path)

# loat target img fom tmp file
img_path: Path = Path("src/model/Attack_data/000_1_0003_1_j.png")
img: Image = Image.open(img_path)  # type: ignore

# load the model
model: simpleCNN = load_cnn_model(
    Path("src/model/trained_CNN.pth"),
    58,
    device=device,
)

# loat the label from the file name

label = int(img_path.name.split("_")[0])
label = int(label)
label: Tensor = torch.tensor([label])

# apply a patch to the image
img_with_patch = paste_rotated_image_with_alpha_at_center(
    background_Image=img,
    overlay_Image=batterfly_img,
    center_position_ratio=PatchCenter,
    size_ratio=patch_size_ratio,
    angle_ratio=patch_angle_ratio,
)
img_with_patch.save("test.png")

# predict the label
img_tensor: Tensor = custom_transform(img_with_patch)
criterion = torch.nn.CrossEntropyLoss()
output = model(img_tensor)
print(f"predict:{torch.argmax(output)}")
print(f"Answer: {label.item()}")
# loss = criterion(output,label)
# loss = loss.detach()
