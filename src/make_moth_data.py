import pathlib
import sys
from PIL import Image


def get_pixels(img):
    pixels = list(img.getdata())
    return pixels


def main():
    img_dir = pathlib.Path("data/images")
    mask_data_dir = pathlib.Path("data/segmentations")
    img_path_list = list(img_dir.iterdir())
    mask_path_list = list(mask_data_dir.iterdir())

    first_img_path = img_path_list[0]
    # first_img_path = (
    #     "/Users/shuhe/Natural_Adversarial_Patch/data/images/0010001.png"
    # )

    first_img_mask_path = mask_path_list[0]

    first_img = Image.open(first_img_path).convert("RGBA")
    first_img_mask = Image.open(first_img_mask_path).convert("L")

    first_img_resized = first_img.resize((256, 256))
    print(first_img_path)
    first_img_resized.show()

    sys.exit()

    new_data = []
    for i, item in enumerate(first_img):
        alpha = 255 if first_img_mask[i] == 2 else 0  # 2なら不透明、0なら透明
        new_data.append((item[0], item[1], item[2], alpha))


if __name__ == "__main__":
    main()
