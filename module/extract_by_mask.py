from PIL import Image
from pathlib import Path


def extract_by_mask(img_path: Path, mask_img_path: Path) -> Image:
    image = Image.open(img_path).convert("RGBA")
    mask = Image.open(mask_img_path).convert("L")
    cropped = Image.composite(
        image, Image.new("RGBA", image.size, (255, 255, 255, 0)), mask
    )
    return cropped


def get_path_list(img_dir: Path):
    """ディレクトリを指定すると、そのディレクトリ内のファイルのパスを返す関数"""
    path_list = list(img_dir.iterdir())
    return path_list
