import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


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
        if Path(img_path).suffix.lower() == '.png':
            if Path(mask_path).suffix.lower() == '.png':                
                cropped = extract_by_mask(img_path, mask_path)
                cropped.save("data/cropped/" + img_path.name)


if __name__ == "__main__":
    main()
