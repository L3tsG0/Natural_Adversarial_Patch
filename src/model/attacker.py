from cProfile import label
from torchvision import transforms as transforms
from pathlib import Path
from requests import patch
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model.cnn import simpleCNN
from src.traffic_data.dataset import CustomDataset
from PIL import Image
from torch import Tensor
from tqdm import tqdm
import torch.utils.tensorboard
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from typing import Final
from src.model.evaluation import load_model
from src.UI.interface import AEengine_interface
import os
import random

def paste_with_alpha(background, overlay, position=(0, 0)):
    """
    Paste the overlay image with alpha channel onto the background image at the specified position.
    """
    # Create a mask using the alpha channel of the overlay image
    mask = overlay.convert("L")

    # Binarize the image using the threshold
    binarized_image = mask.point(lambda p: 255 if p > 0.2 else 0, mode="1")

    # Paste the overlay onto the background using the mask
    background.paste(overlay, position, binarized_image)
    return background


def paste_rotated_image_with_alpha_at_center(
    background_Image,
    overlay_Image,
    center_position_ratio=(0.1, 0.1),
    max_angle=360,
    size_ratio=0.2,
    angle_ratio=0.2,
):
    """
    Paste a rotated version of the overlay image (with alpha channel) onto the background image
    at the specified center position.

    Args:
    - background_path (str): Path to the background image.
    - overlay_path (str): Path to the overlay image.
    - center_position (tuple): Center position where the overlay image will be pasted.
    - max_angle (int): Maximum rotation angle for the overlay image.

    Returns:
    - Image: Combined image.
    """

    # 画像の定義
    background = background_Image
    overlay = overlay_Image

    # サイズをbackgroundに合わせる
    size = background.size
    overlay = overlay.resize(
        (int(overlay.size[0] * size_ratio), int(overlay.size[1] * size_ratio))
    )

    # Rotate the overlay image
    angle = angle_ratio * 360
    overlay_rotated = overlay.rotate(angle, expand=True, resample=Image.BICUBIC)

    # Calculate the position to paste the overlay so that its center aligns with the specified center position
    overlay_width, overlay_height = overlay_rotated.size
    center_position = (
        int(center_position_ratio[0] * size[0]),
        int(center_position_ratio[1] * size[1]),
    )
    position = (
        center_position[0] - overlay_width // 2,
        center_position[1] - overlay_height // 2,
    )

    # Paste the rotated overlay onto the background considering the alpha channel
    combined_image = paste_with_alpha(background, overlay_rotated, position)

    return combined_image


def CutoutEdge(img_path, margin=0.05):
    """
    Cut the edges according to the parameter of margin.

    """

    im = Image.open(img_path)
    width, height = im.size

    # Calculate the bounding box considering the margins
    left: float = margin * width
    upper: float = margin * height
    right: float = width - margin * width
    lower: float = height - margin * height

    # Crop the image using the bounding box
    cropped_image = im.crop((int(left), int(upper), int(right), int(lower)))

    return cropped_image


def load_cnn_model(
    path: Path, num_classes: int = 58, device: torch.device = None
) -> simpleCNN:
    model = simpleCNN(num_classes)# モデル構造の定義
    model.load_state_dict(torch.load(map_location=device, f=path))# 重みファイルのロード
    return model




def compute_cnn_test_accuracy(device: torch.device):
    test_dataset = CustomDataset(
        Path("data/traffic_sign/success_result"), is_train=False
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

    model = load_cnn_model(
        Path("src/model/trained_CNN.pth"),
        test_dataset.num_classes,
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


def custom_transform(img):
    """画像の前処理"""
    img = img.resize(
        (224, 224)
    )  # 参照: https://www.kaggle.com/code/boulahchichenadir/cnn-classification
    img = transforms.ToTensor()(img)
    img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
    return img


def calc_reward(
    img, label_, overlay, PatchCenters: list[list[float]], batch_size, patch_size_ratio,target_model=None
) -> Tensor:
    reward_: list[float] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = target_model

    label: Tensor = torch.tensor([label_])

    print(label)

    for i in range(batch_size):
        # patch apply
        PatchCenter = (PatchCenters[i][0], PatchCenters[i][1])
        patch_angle_ratio = PatchCenters[i][2]

        img_with_patch = paste_rotated_image_with_alpha_at_center(
            img,
            overlay,
            center_position_ratio=PatchCenter,
            size_ratio=patch_size_ratio,
            angle_ratio=patch_angle_ratio,
        )


        # import img from tmp file

        img_tensor = custom_transform(img_with_patch)
        criterion = torch.nn.CrossEntropyLoss()
        output = model(img_tensor)
        loss = criterion(output, label)
        loss = loss.detach()

        reward_.append(loss)

    return torch.stack(reward_)

interface: AEengine_interface = None
def set_interface(arg: AEengine_interface):
    global interface
    interface = arg

usecsv = True
csv_path = ""
def set_usecsv(setting: bool, _csv_path: str):
    global usecsv, csv_path
    usecsv = setting
    csv_path = _csv_path

def dump_patched_image_and_predict_label(position: Tensor, target_model_path:str, target_image_path: str, patch_image_path: str,):
    global usecsv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setting from the result of PEPG
    # position = torch.tensor([0.0000, 0.5173, 0.0000])

    PatchCenter: tuple[Tensor, Tensor] = (position[0], position[1])
    patch_angle_ratio: float = float(position[2])
    patch_size_ratio: float = 0.3


    # load batterfly img
    #batterfly_img_path = Path("src/model/Attack_data/0010001.png")
    batterfly_img_path = Path(patch_image_path)
    batterfly_img: Image.Image = CutoutEdge(batterfly_img_path)

    # loat target img fom tmp file
    #img_path: Path = Path("src/model/Attack_data/000_1_0003_1_j.png")
    img_path: Path = Path(target_image_path)
    img: Image.Image = Image.open(img_path)  # type: ignore

    # load the model
    model: torch.nn.Module = load_model(
        #Path("src/model/trained_CNN.pth"),
        Path(target_model_path),
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

    if not os.path.exists("export"):
        os.mkdir("export")
    i = 0
    while True:
        patch_filename = f"export/patched_image_{i}.png"
        if os.path.exists(patch_filename):
            i += 1
        else:
            break

    img_with_patch.save(patch_filename)

    # predict the label
    img_tensor: Tensor = custom_transform(img_with_patch)
    criterion = torch.nn.CrossEntropyLoss()
    output: Tensor = model(img_tensor.to(device))
    predict_label:int = int(torch.argmax(output).item())
    print(f"predict:{predict_label}")
    print(f"Answer: {label.item()}")


    return patch_filename, predict_label, label.item()

stop = False
def stop_cycles():
    global stop
    stop = True



def main(target_model_path:str, target_image_path: str, patch_image_path: str, learning_cycles: int):
    global stop, interface

    # define the patch size
    patch_size_ratio = 0.2

    # settings for PEPG
    N: Final[int] = 3
    batch: Final[int] = 20
    now = datetime.now()
    num_iter = learning_cycles
    lr = 0.002

    # setting for log of PEPG
    dir_name: str = now.strftime("runs/%Y%m%d_%H%M%S")
    writer = SummaryWriter(dir_name)

    max_rewards: list[float] = []
    baseline_history: list[float] = []
    highest_reward:float = float("-inf")
    best_patch_position = None

    # define initial state
    xy_mu_init = torch.rand(3)
    xy_sigma_init = torch.tensor([2.0 for i in range(N)])
    xy_mu: Tensor = xy_mu_init.clone().detach()
    xy_sigma: Tensor = xy_sigma_init.clone().detach()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # open the target image
    # todo:UI から読み込めるようにする
    img_path: Path = Path(target_image_path)
    img: Image = Image.open(img_path)  # type: ignore
    img_tensor = custom_transform(img)

    # open the batterfly image
    # todo:UIから画像を読み込めるようにする
    
    # batterfly_imgs = []
    # if "@" in patch_image_path:
    #     tmp = patch_image_path.split("@")
        
    #     for i in range(0, 64):
    #         batterfly_imgs.append(CutoutEdge(Path(tmp[0] + str(i) + tmp[1])))
    # else: 
    #     batterfly_imgs.append(CutoutEdge(Path(tmp[0] + str(i) + tmp[1])))   
    batterfly_img_path = Path(patch_image_path) #パッチのパス
    batterfly_img: Image = CutoutEdge(batterfly_img_path)

    # get ground truth label from the file name.
    # ファイル名から，正解ラベル側ことが前提(データセット依存)
    # todo:自分で与えるか，(攻撃加える前の)画像のモデルの出力を正解とする

    label = int(img_path.name.split("_")[0])
    label = int(label)
    label: Tensor = torch.tensor([label])

    # load the model

    model = torch.load(target_model_path)

    criterion = torch.nn.CrossEntropyLoss()

    # compute loss
    output = model(img_tensor)
    loss = criterion(output, label)

    # for log
    with open("PatchApply.txt", "a") as f:
        f.write(f"PatchSize:{patch_size_ratio}\n")
        f.write(f"xy_mu_init:{xy_mu_init}\n")
        f.write(f"xy_sigma_init:{xy_sigma_init}\n")
        f.write(f"lr:{lr}\n")
        f.write(f"num_iter:{num_iter}\n")
        f.write(f"batch:{batch}\n")
        f.write(f"PatchNum:{N/2}\n")

    # PEPGアルゴリズムを用いたパッチ貼り付け位置の最適化
    with torch.no_grad():
        for iter in tqdm(range(num_iter)):
            if stop is True:
                stop = False
                return highest_reward,best_patch_position
            rewards: Tensor = torch.tensor([])
            mu_grad: Tensor = torch.zeros(N)
            sigma_grad: Tensor = torch.zeros(N)
            xy_combined: Tensor = torch.zeros([])

            # generate noise from Normal distribution

            noise = torch.tensor(
                [
                    torch.normal(mean=0.0, std=s, size=(batch,)).tolist()
                    for s in xy_sigma
                ]
            )

            # symmetric sampling

            xy_positive: Tensor = (xy_mu + noise.T).clip(0.0, 1.0)
            xy_negative: Tensor = (xy_mu - noise.T).clip(0.0, 1.0)

            xy_combined = torch.cat((xy_positive, xy_negative))

            # calc reward
            # batterfly_img = batterfly_imgs[random.randint(0, 63)]
            r_positive: Tensor = calc_reward(
                img=img,
                label_=label,
                overlay=batterfly_img,
                PatchCenters=xy_positive,
                batch_size=batch,
                patch_size_ratio=patch_size_ratio,
                target_model = model,
            )
            r_negative: Tensor = calc_reward(
                img=img,
                label_=label,
                overlay=batterfly_img,
                PatchCenters=xy_negative,
                batch_size=batch,
                patch_size_ratio=patch_size_ratio,
                target_model = model,
            )

            # select max_reward for regularization

            combine_reward: Tensor = torch.cat((r_positive, r_negative))

            max_reward: float = torch.max(combine_reward).item()
            max_reward_index: int = torch.argmax(combine_reward).item() 
            max_rewards.append(max_reward)
            if highest_reward < max_reward:
                highest_reward = max_reward
                best_patch_position = xy_combined[max_reward_index].tolist()

                # 最大報酬が更新されたとき、最大報酬のパッチを出力し、UIに反映
                patch_filename, predict_label, answer_label = dump_patched_image_and_predict_label(\
                    torch.tensor(best_patch_position), \
                    target_model_path, target_image_path, \
                    patch_image_path)
                if interface:
                    interface.ui_end_1optimize_iteration(patch_filename, highest_reward, predict_label, answer_label)

            with open("PatchApply.txt", "a") as f:
                f.write(f"max_reward:{max_reward}\n")
                f.write(f"patch_positions:{xy_combined[max_reward_index]}\n")
                f.write(f"max_reward:{max_reward}\n")
                f.write(f"patch_positions:{xy_combined[max_reward_index]}\n")

            writer.add_scalar("max_reward", max_reward, iter)

            # calc baseline for regularization

            rewards: Tensor = torch.cat((rewards, combine_reward))  # change
            baseline: float = torch.mean(rewards).item()
            baseline_history.append(baseline)

            # calc update of hyper-params
            r_t: Tensor = (r_positive - r_negative) / ((-1) * (r_negative + r_positive))
            r_s: Tensor = ((r_positive + r_negative) / 2.0 - baseline) / (
                (-1) * baseline
            )

            mu_grad = 2 * torch.mm(noise, r_t.float().unsqueeze(1))
            sigma_grad = torch.mm(
                ((noise.float().T * 2 - xy_sigma.float() ** 2) / xy_sigma).T,
                r_s.float().unsqueeze(1),
            )

            writer.add_scalar("mu_grad", mu_grad[0][0], iter)
            writer.add_scalar("sigma_grad", sigma_grad[0][0], iter)
            writer.add_scalar("baseline", baseline, iter)

            # update hyper-params

            xy_mu += lr * mu_grad[:, 0]
            xy_sigma = xy_sigma.float()
            xy_sigma += lr * sigma_grad[:, 0]
            xy_sigma = xy_sigma.clip(0.5)

            writer.add_scalar("xy_mu", xy_mu[0], iter)
            writer.add_scalar("xy_sigma", xy_sigma[0], iter)      

            # 学習の進捗をUIに反映
            if interface:
                interface.set_convergence("", (iter+1)/num_iter)

    return highest_reward,best_patch_position

if __name__ == "__main__":
    max_reward,best_position = main("src/model/trained_CNN.pth", \
                                    "src/model/Attack_data/000_1_0003_1_j.png", \
                                    "src/model/Attack_data/0010001.png", \
                                    200)
    print(f"max_reward:{max_reward}")
    print(f"Best Patch Position: {best_position}")
