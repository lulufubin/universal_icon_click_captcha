from decimal import Decimal
import hashlib
from pathlib import Path
import random
from typing import Dict, Tuple, Union
import uuid

from PIL import Image
import numpy as np


class Datasets:
    def __init__(
        self,
        background_images_path: Union[Path, str],
        icon_images_path: Union[Path, str],
        save_path: Union[Path, str],
        bg_image_ext: str = "jpg",
        icon_image_ext: str = "png",
    ):
        """
        Args:
            background_images_path (Union[Path, str]): 背景图文件路径,JPG为主
            icon_images_path (Union[Path, str]): 图标文件路径
        """
        if isinstance(background_images_path, str):
            background_images_path = Path(background_images_path)
        if isinstance(icon_images_path, str):
            icon_images_path = Path(icon_images_path)

        if isinstance(save_path, str):
            save_path = Path(save_path)

        self.background_images_path = list(
            background_images_path.glob(f"*.{bg_image_ext}")
        )
        random.seed(12)
        random.shuffle(self.background_images_path)
        self.icon_images_path = list(icon_images_path.glob(f"*.{icon_image_ext}"))
        random.seed(13)
        random.shuffle(self.icon_images_path)
        random.seed(None)
        self.save_path = save_path

    def _get_image_hash(self, image_path: Path):
        return hashlib.md5(open(image_path, "rb").read()).hexdigest()

    def generate_by_yolo(
        self,
        hash_classidx_dict: Dict[str, int],
        padding: int = 5,
        group_icon_num: int = 3,
        group_icon_auto_fill: bool = True,
        img_enhance: bool = True,
        random_rotate: bool = False,
        random_filp: bool = False,
        random_color: bool = False,
        target_color: Tuple[int, int, int] = (0, 0, 0),
        tolerance: int = 30,
    ):
        """
        生成yolo数据集
        Args:
            hash_classidx_dict   (Dict[str, int]): 图标文件的哈希值对应的类别索引。
            padding              (int)           : 图标摆放位置距离边框的距离。Defaults:5
            group_icon_num       (int)           : 每张背景图上显示图标的数量。Defaults:3
            group_icon_auto_fill (bool)          : 当图标数量与group_icon_num不能完美分割时，随机抽取图标进行填充。Defaults: True
            img_enhance          (bool)          : 图片增强。Defaults:True
            train_name           (str)           : 训练集名称
        """
        save_image_path = self.save_path.joinpath("images")
        save_label_path = self.save_path.joinpath("labels")
        if not save_image_path.exists():
            save_image_path.mkdir(parents=True, exist_ok=True)
        if not save_label_path.exists():
            save_label_path.mkdir(parents=True, exist_ok=True)

        bg_images_path = self.background_images_path
        icon_images_path = [
            self.icon_images_path[i : i + group_icon_num]
            for i in range(0, len(self.icon_images_path), group_icon_num)
        ]

        # 检查最后一张图片是否满足要求
        if group_icon_auto_fill and len(icon_images_path[-1]) < group_icon_num:
            for _ in range(group_icon_num - len(icon_images_path)):
                icon_images_path[-1].append(
                    self.icon_images_path[random.randint(0, len(self.icon_images_path))]
                )

        current_index = 0
        for bg_image_path in bg_images_path:
            bg_image = Image.open(bg_image_path)
            bg_width, bg_height = bg_image.size
            regions_x = bg_width // 3

            group_icon_paths = icon_images_path[current_index]

            yolo_annotations = []
            for index, icon_image_path in enumerate(group_icon_paths):
                icon_image = Image.open(icon_image_path).convert("RGBA")
                if img_enhance:
                    icon_image = self._perform_icon_image_augmentation(
                        icon_image,
                        random_rotate,
                        random_filp,
                        random_color,
                        target_color,
                        tolerance,
                    )
                icon_width, icon_height = icon_image.size
                x = random.randint(
                    padding + (index * regions_x),
                    (index + 1) * regions_x - icon_width,
                )
                y = random.randint(
                    padding,
                    bg_height - icon_height,
                )
                bg_image.paste(icon_image, (x, y), icon_image)

                # 图标位置转成yolo训练格式
                x_center = Decimal(x + icon_width / 2)
                y_center = Decimal(y + icon_height / 2)
                x_center_norm = Decimal(x_center / bg_width)
                y_center_norm = Decimal(y_center / bg_height)
                width_norm = Decimal(icon_width / bg_width)
                height_norm = Decimal(icon_height / bg_height)
                hash = self._get_image_hash(icon_image_path)
                if hash not in hash_classidx_dict:
                    raise ValueError(f"找不到hash值所对应的图标类型:\t{hash}")
                yolo_annotations.append(
                    [
                        str(hash_classidx_dict[hash]),
                        str(x_center_norm.quantize(Decimal("0.000000"))),
                        str(y_center_norm.quantize(Decimal("0.000000"))),
                        str(width_norm.quantize(Decimal("0.000000"))),
                        str(height_norm.quantize(Decimal("0.000000"))),
                    ]
                )
            # 保存相关数据
            filename = uuid.uuid4()
            with open(
                save_label_path.joinpath(f"{filename}.txt"), "w", encoding="utf-8"
            ) as f:
                for yolo_data in yolo_annotations:
                    f.write(" ".join(yolo_data) + "\n")
                f.close()
            bg_image.save(save_image_path.joinpath(f"{filename}.jpg"),format="JPEG")
            current_index += 1
            if current_index >= len(icon_images_path):
                current_index = 0

    def _perform_icon_image_augmentation(
        self,
        image: Image.Image,
        random_rotate: bool = False,
        random_filp: bool = False,
        random_color: bool = False,
        target_color: Tuple[int, int, int] = (0, 0, 0),
        tolerance: int = 30,
    ) -> Image.Image:
        """
        对图标就行预处理

        - 随机旋转
        - 随机水平翻转
        - 随机垂直翻转

        Args:
            image        (Image.Image)       : 需要预处理的图标
            random_rotate(bool)              : 随机旋转
            random_filp  (bool)              : 随机翻转
            random_color (bool)              : 随机颜色
            target_color (Tuple[int,int,int]): 需要修改的颜色
            tolerance    (int)               : 颜色容差，控制颜色匹配的精确度

        Returns:
            Image.Image: 预处理后的图标
        """
        if random_color and random.random() > 0.5:
            image = self._change_icon_color(
                image,
                target_color,
                (
                    random.randint(1, 255),
                    random.randint(1, 255),
                    random.randint(1, 255),
                ),
                tolerance,
            )
        if random_rotate and random.random() > 0.5:
            image = image.rotate(
                random.randint(0, 360), expand=True, resample=Image.Resampling.BICUBIC
            )
        if random_filp and random.random() > 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random_filp and random.random() > 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        return image

    def _change_icon_color(
        self,
        image: Image.Image,
        target_color: Tuple[int, int, int],
        replace_color: Tuple[int, int, int],
        tolerance: int = 30,
    ) -> Image.Image:
        """
        改变图标的颜色

        Args:
            image (Image.Image): 需要处理的图标
            target_color (Tuple[int, int, int]): 需要修改的RGB颜色值
            replace_color (Tuple[int, int, int]): 替换成指定颜色的RGB值
            tolerance (int, optional): 颜色容差. Defaults to 30.

        Returns:
            Image.Image: 处理后的图标
        """
        data = np.array(image)

        red, green, blue, alpha = data.T

        r_mask = (red >= target_color[0] - tolerance) & (
            red <= target_color[0] + tolerance
        )
        g_mask = (green >= target_color[1] - tolerance) & (
            green <= target_color[1] + tolerance
        )
        b_mask = (blue >= target_color[2] - tolerance) & (
            blue <= target_color[2] + tolerance
        )

        mask = r_mask & g_mask & b_mask

        data[..., :-1][mask.T] = replace_color

        return Image.fromarray(data)
