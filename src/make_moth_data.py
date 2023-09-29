import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def main():
    base_dir = Path("../data/leedsbutterfly")
    images_path = base_dir / "images"
    masks_path = base_dir / "segmentations"

    # それぞれのディレクトリ内のファイル名のリストを取得
    # 拡張子以外: 拡張子を含めたファイル名の辞書形式で保存
    # 例: {"image1": "image1.png"}
    images_files: dict[str, Path] = {
        f.stem: f
        for f in images_path.iterdir()
        if f.is_file() and f.suffix.lower() == ".png"
    }
    masks_files: dict[str, Path] = {
        f.stem.replace("_seg0", ""): f
        for f in masks_path.iterdir()
        if f.is_file()
    }

    # もしディレクトリが存在しなかったら作る
    cropped_dir: Path = base_dir / "cropped"
    if cropped_dir.exists is False:
        print(f"{cropped_dir} が存在しないので作成します")
        os.mkdir(cropped_dir)

    # 両リストの共通のファイル名を基にペアを作成
    paired_files: list[tuple[Path, Path]] = [
        (images_files[name], masks_files[name])
        for name in images_files
        if name in masks_files
    ]

    for img_path_, mask_img_path_ in tqdm(paired_files, desc="データセット作成しています"):

        # 画像を読み込む
        mask_img = Image.open(mask_img_path_).convert("L")
        img_np= np.array(mask_img)

        img = Image.open(img_path_).convert("RGB")

        cropped_img = Image.composite(img, Image.new("RGBA", img.size, (255, 255, 255, 0)),mask_img)
        

        # モルフォロジー変換を適用してノイズを除去
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)

        # 最大の連結成分を検出
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8)
        largest_label = np.argmax(stats[1:, 4]) + 1  # 0は背景のため除外
        binary_cleaned = np.where(labels == largest_label, 255, 0).astype(
            "uint8"
        )

        # 切り取る領域の座標を決定
        non_zero_coords = np.column_stack(np.where(binary_cleaned > 0))
        min_y, min_x = non_zero_coords.min(axis=0) - 10
        max_y, max_x = non_zero_coords.max(axis=0) + 10

        # 画像を切り取る
        cropped_img = cropped_img.crop((min_x, min_y, max_x, max_y))
        
        cropped_img.save(Path(cropped_dir) / img_path_.name)


if __name__ == "__main__":
    main()
    