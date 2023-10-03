from cProfile import label

from pathlib import Path
from requests import patch
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model.cnn import simpleCNN
from src.traffic_data.dataset import CustomDataset
from PIL import Image
from torch import Tensor
# import tensorboard
from tqdm import tqdm
import torch.utils.tensorboard
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from typing import Final


def paste_with_alpha(background, overlay, position=(0, 0)):
    """
    Paste the overlay image with alpha channel onto the background image at the specified position.
    """
    # Create a mask using the alpha channel of the overlay image
    mask = overlay.convert("L")


    
    # Binarize the image using the threshold
    binarized_image = mask.point(lambda p: 255 if p > 0.2 else 0, mode='1')
    

    
    # Paste the overlay onto the background using the mask
    background.paste(overlay, position, binarized_image)
    return background

def paste_rotated_image_with_alpha_at_center(background_Image, overlay_Image, center_position_ratio=(0.1, 0.1), max_angle=360,ratio = 0.2,angle_ratio = 0.2):
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


    # Open the images
    # background = Image.open(background_path)
    angle = angle_ratio*360
    background = background_Image
    overlay = overlay_Image
    # サイズをbackgroundに合わせる
    size = background.size
    
    # print((int(center_position_ratio[0]*size[0]),int(center_position_ratio[1]*size[1])))
    center_position = (int(center_position_ratio[0]*size[0]),int(center_position_ratio[1]*size[1]))
    overlay = overlay.resize(tuple(size))
    
    overlay = overlay.resize((int(overlay.size[0]*ratio),int(overlay.size[1]*ratio)))
    
    # Rotate the overlay image by a random angle
    # angle = torch.randint(0, max_angle, (1,)).item()
    overlay_rotated = overlay.rotate(angle, expand=True, resample=Image.BICUBIC)
    
    # Calculate the position to paste the overlay so that its center aligns with the specified center position
    overlay_width, overlay_height = overlay_rotated.size
    position = (center_position[0] - overlay_width // 2, center_position[1] - overlay_height // 2)
    
    # Paste the rotated overlay onto the background considering the alpha channel
    combined_image = paste_with_alpha(background, overlay_rotated, position)
    
    return combined_image

# Again, for demonstration purposes, we won't run the function as we don't have access to image files here.
# But you can use the `paste_rotated_image_with_alpha_at_center` function in your environment where you have images.

def CutoutEdge(img_path,margin = 0.05):
    im = Image.open(img_path)
    im_shape = im.size
    width, height =  im.size
    # Calculate the bounding box considering the margins
    left = margin * width
    upper = margin * height
    right = width - margin * width
    lower = height - margin * height


    # Crop the image using the bounding box
    cropped_image = im.crop((left, upper, right, lower))

    # cropped_image.save("cropped_image.png")
    return cropped_image

def load_cnn_model(
    path: Path, num_classes: int = 58, device: torch.device = None
) -> simpleCNN:
    model = simpleCNN(num_classes)
    model.load_state_dict(torch.load(map_location=device, f=path))
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

from torchvision import transforms as transforms
def custom_transform(img):
    """画像の前処理"""
    img = img.resize(
        (224, 224)
    )  # 参照: https://www.kaggle.com/code/boulahchichenadir/cnn-classification
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )(img)
    return img




def calc_reward(img,label_,overlay,PatchCenters: list[list[float]],batch_size,patch_size_ratio) -> Tensor:
    reward_: list[float] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: simpleCNN = load_cnn_model(
        Path("src/model/trained_CNN.pth"),
        58,
        device=device,
    )

    
    label: Tensor = torch.tensor([label_])
    
    print(label)

    for i in range(batch_size):
        # patch apply
        PatchCenter = (PatchCenters[i][0],PatchCenters[i][1])
        patch_angle_ratio = PatchCenters[i][2]

        

        img_with_patch = paste_rotated_image_with_alpha_at_center(img,overlay,center_position_ratio= PatchCenter,ratio = patch_size_ratio,angle_ratio=patch_angle_ratio)

        # savefig in tmpfile
        # img_with_patch.save(Path(f"tmp/output{i}.jpg"))

    # import img fom tmp file

        img_tensor = custom_transform(img_with_patch)
        criterion=torch.nn.CrossEntropyLoss()
        output = model(img_tensor)
        loss = criterion(output,label)
        loss = loss.detach()



    

        

        reward_.append( loss)

    return torch.stack(reward_)




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # compute_cnn_test_accuracy(device)

    #  読み込む画像を決める

    
    
    img_path: Path = Path("src/model/Attack_data/000_1_0003_1_j.png")
    img: Image = Image.open(img_path) # type: ignore

    model: simpleCNN = load_cnn_model(
        Path("src/model/trained_CNN.pth"),
        58,
        device=device,
    )
    label = int(img_path.name.split("_")[0])
    label = int(label)
    label: Tensor = torch.tensor([label])
    patch_size_ratio = 0.2
    print(label)
    
    # img.save("uni.png")
    img_tensor = custom_transform(img)
    criterion=torch.nn.CrossEntropyLoss()
    output = model(img_tensor)
    loss = criterion(output,label)
    print(loss)

    # 町の画像のディレクトリ指定
    batterfly_img_path = Path("src/model/Attack_data/0010001.png")
    # batterfly_img = Image.open(batterfly_img_path)

    batterfly_img = CutoutEdge(batterfly_img_path)
    # batterfly_img.save("test.png")
    # print(img.size())

    

    # combined_image.save("test.png")

    




    



    # PEPGアルゴリズム

    # combined_image = paste_rotated_image_with_alpha_at_center(img,batterfly_img, max_angle=360,center_position_ratio=(0.5, 0.5),angle=140,ratio=0.7)
        # パッチの大きさは固定
        # 位置(x,y),と回転を最適化
        # ロスの数は固定

    # 初期設定
    N : Final[int] = 3
    batch:Final[int] = 20
    now = datetime.now()
    num_iter = 100
    
    lr = 0.002

    dir_name = now.strftime("runs/%Y%m%d_%H%M%S")

    writer = SummaryWriter(dir_name)


    max_rewards: list[float] = []

    xy_mu_init = torch.rand(3)
    xy_sigma_init = torch.tensor([2. for i in range(N)])

    xy_mu: Tensor = torch.tensor(xy_mu_init)
    xy_sigma: Tensor = torch.tensor(xy_sigma_init)

    baseline_history: list[float] = []

    with open("PatchApply.txt", "a") as f:
        f.write(f"PatchSize:{patch_size_ratio}\n")
        f.write(f"xy_mu_init:{xy_mu_init}\n")
        f.write(f"xy_sigma_init:{xy_sigma_init}\n")
        f.write(f"lr:{lr}\n")
        f.write(f"num_iter:{num_iter}\n")
        f.write(f"batch:{batch}\n")
        f.write(f"PatchNum:{N/2}\n")

    with torch.no_grad():
        for iter in tqdm(range(num_iter)):
            rewards: Tensor = torch.tensor([])
            mu_grad: Tensor = torch.zeros(N)
            sigma_grad: Tensor = torch.zeros(N)
            xy_combined: Tensor = torch.zeros([])

            # generate noise from Normal distribution

            noise = torch.tensor(
                [torch.normal(mean=0.0, std=s, size=(batch,)).tolist() for s in xy_sigma]
            )

            # symmetric sampling

            xy_positive: Tensor = (xy_mu + noise.T).clip(0.,1.)
            xy_negative: Tensor = (xy_mu - noise.T).clip(0.,1.)

            xy_combined = torch.cat((xy_positive, xy_negative))

            # print(xy_positive)

            # calc reward

            r_positive: Tensor = calc_reward(
                img=img,label_=label,overlay=batterfly_img,PatchCenters=xy_positive,batch_size=batch,patch_size_ratio=patch_size_ratio
            )
            r_negative: Tensor = calc_reward(
                img = img,label_ = label,overlay = batterfly_img,PatchCenters=xy_negative,batch_size=batch,patch_size_ratio = patch_size_ratio
            )


            # select max_reward for regularization

            combine_reward: Tensor = torch.cat((r_positive, r_negative))

            max_reward: float = torch.max(combine_reward).item() 
            max_reward_index: int = torch.argmax(combine_reward).item()
            max_rewards.append(max_reward)

            with open("PatchApply.txt", "a") as f:
                f.write(f"max_reward:{max_reward}\n")
                f.write(f"patch_positions:{xy_combined[max_reward_index]}\n")


            with open("PatchApply.txt", "a") as f:
                f.write(f"max_reward:{max_reward}\n")
                f.write(f"patch_positions:{xy_combined[max_reward_index]}\n")

            writer.add_scalar("max_reward", max_reward, iter)

            # calc baseline fro regularization

            rewards: Tensor = torch.cat((rewards, combine_reward))# change

            baseline: float = torch.mean(rewards).item()

            baseline_history.append(baseline)
            

            # calc reward

            r_t: Tensor = (r_positive - r_negative)/((-1)*(r_negative+r_positive))
            r_s :Tensor = ((r_positive+r_negative)/2.-baseline)/((-1)*baseline)


            

            mu_grad = 2 * torch.mm(noise, r_t.float().unsqueeze(1))
            sigma_grad = torch.mm(((noise.float().T*2 - xy_sigma.float()**2) / xy_sigma).T, r_s.float().unsqueeze(1))

            writer.add_scalar("mu_grad", mu_grad[0][0], iter)
            writer.add_scalar("sigma_grad", sigma_grad[0][0], iter)
            writer.add_scalar("baseline", baseline, iter)

            

            # update hyper-params

        with torch.no_grad():
            xy_mu += lr * mu_grad[:, 0]
            xy_sigma = xy_sigma.float()
            xy_sigma +=lr* sigma_grad[:, 0]
            xy_sigma = xy_sigma.clip(0.5)

            # writer.add_scalar("xy_mu", xy_mu[0], iter)
            # writer.add_scalar("xy_sigma", xy_sigma[0], iter)



if __name__ == "__main__":
    main()
