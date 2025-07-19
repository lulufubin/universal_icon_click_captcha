import hashlib
from pathlib import Path
from typing import Dict
from datasets import Datasets

# 图标名称
classes = [
  "蚂蚁","青蛙"
]


def get_file_hash(file: Path) -> str:
    return hashlib.md5(open(file, "rb").read()).hexdigest()


def get_hash_classes_dict(icon_paths):
    icons = Path(icon_paths).glob("*.png")
    data: Dict[str, int] = {}
    for icon in icons:
        filename = icon.stem
        hash = get_file_hash(icon)
        data[hash] = classes.index(filename)
    return data


def main():
    # 背景图路径
    bg_paths = ""
    # 图标路径
    icon_paths = ""
    # 数据集保存位置
    save_path = ""
    datasets = Datasets(
        background_images_path=bg_paths,
        icon_images_path=icon_paths,
        save_path=save_path,
    )
    datasets.generate_by_yolo(
        get_hash_classes_dict(icon_paths),
        random_color=True,
        random_filp=True,
        random_rotate=True,
        target_color=(0, 0, 0),
    )


if __name__ == "__main__":
    main()
