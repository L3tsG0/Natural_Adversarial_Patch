from PIL import Image
from pathlib import Path


def extract_by_mask(img_path:Path,mask_img_path:Path)->Image:
    image = Image.open(img_path).convert("RGBA")
    mask = Image.open(mask_img_path).convert("L")
    cropped = Image.composite(image, Image.new('RGBA', image.size, (255, 255, 255,0)), mask)
    return cropped