import torch
from pathlib import Path
from src.model.attacker import (
    CutoutEdge,
    custom_transform,
    paste_rotated_image_with_alpha_at_center,
)
from src.model.evaluation import load_model
from PIL import Image
from torch import Tensor

usecsv = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setting from the result of PEPG
position = torch.tensor([0.0000, 0.5173, 0.0000])

PatchCenter: tuple[Tensor, Tensor] = (position[0], position[1])
patch_angle_ratio: float = float(position[2])
patch_size_ratio: float = 0.3


# load batterfly img
batterfly_img_path = Path("src/model/Attack_data/0010001.png")
batterfly_img: Image.Image = CutoutEdge(batterfly_img_path)

# loat target img fom tmp file
img_path: Path = Path("src/model/Attack_data/000_1_0003_1_j.png")
img: Image.Image = Image.open(img_path) 

# load the model
model: torch.nn.Module = load_model(
    Path("src/model/trained_CNN.pth"),
    device=device,
)

# loat the label from the file name

if usecsv is True:
    label:Tensor = torch.tensor([int(img_path.name.split("_")[0])])
else:
    label: Tensor = torch.argmax(model(img))

# apply a patch to the image
img_with_patch:Image.Image = paste_rotated_image_with_alpha_at_center(
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
output: Tensor = model(img_tensor)
predict_label:int = int(torch.argmax(output).item())
print(f"predict:{predict_label}")
print(f"Answer: {label.item()}")
