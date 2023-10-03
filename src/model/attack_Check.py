from torch import tensor
import torch
from pathlib import Path
from src.model.attacker import CutoutEdge, custom_transform, paste_rotated_image_with_alpha_at_center
from src.model.cnn import simpleCNN
from src.model.evaluation import load_cnn_model
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

position = torch.tensor([0.0000, 0.5173, 0.0000])
PatchCenter = (position[0],position[1])
patch_angle_ratio = position[2]

batterfly_img_path = Path("src/model/Attack_data/0010001.png")
    # batterfly_img = Image.open(batterfly_img_path)
batterfly_img = CutoutEdge(batterfly_img_path)
        
patch_size_ratio = 0.3

        # savefig in tmpfile
        # img_with_patch.save(Path(f"tmp/output{i}.jpg"))

    # import img fom tmp file
img_path: Path = Path("src/model/Attack_data/000_1_0003_1_j.png")
img: Image = Image.open(img_path) # type: ignore

model: simpleCNN = load_cnn_model(
    Path("src/model/trained_CNN.pth"),
    58,
    device=device,
)


label = int(img_path.name.split("_")[0])
label = int(label)
label = torch.tensor([label])
img_with_patch = paste_rotated_image_with_alpha_at_center(img,batterfly_img,center_position_ratio= PatchCenter,ratio = patch_size_ratio,angle_ratio=patch_angle_ratio)
img_with_patch.save("test.png")
img_tensor = custom_transform(img_with_patch)
criterion=torch.nn.CrossEntropyLoss()
output = model(img_tensor)
print(f"predict:{torch.argmax(output)}")
print(f"Answer: {label.item()}")
loss = criterion(output,label)
loss = loss.detach()
