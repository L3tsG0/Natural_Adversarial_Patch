import os
from pathlib import Path
from tqdm import tqdm


from module.extract_by_mask import (
    extract_by_mask,
    get_path_list,
)


def main():
    img_dir = Path("data/images")
    mask_data_dir = Path("data/segmentations")

    img_path_list = get_path_list(img_dir)
    mask_path_list = get_path_list(mask_data_dir)

    if os.path.exists("data/cropped") is False:
        print("data/cropped が存在しないので作成します")
        os.mkdir("data/cropped")

    for img_path, mask_path in zip(
        tqdm(sorted(img_path_list), desc="データセット作成しています"),
        sorted(mask_path_list),
    ):
        cropped = extract_by_mask(img_path, mask_path)
        cropped.save("data/cropped/" + img_path.name)


if __name__ == "__main__":
    main()
