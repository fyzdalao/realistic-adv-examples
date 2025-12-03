#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NPY_DIR = SCRIPT_DIR / "npys"
DEFAULT_PIC_DIR = SCRIPT_DIR / "pics"
IMAGE_FILES = [
    "59.npy",
    "59_adv.npy",
    "59_advv.npy",
]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = DEFAULT_NPY_DIR / path
    return path


def load_image(path: Path) -> np.ndarray:
    img = np.load(path)
    if img.ndim == 3:
        if img.shape[0] in {1, 3}:
            img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = np.clip(img, 0.0, 1.0)
    return img


def main():
    parser = argparse.ArgumentParser(description="显示原图与对抗样本")
    parser.add_argument("--output",
                        default=None,
                        help=f"输出图片路径（默认保存到 {DEFAULT_PIC_DIR} 下并自动命名）")
    parser.add_argument("--times",
                        type=float,
                        default=1.0,
                        help="放大对抗扰动的倍数（orig 保持不变）")
    args = parser.parse_args()

    if not IMAGE_FILES:
        raise ValueError("IMAGE_FILES 列表为空，请在脚本中配置需要展示的 npy 路径。")

    paths = [resolve_path(name) for name in IMAGE_FILES]
    images = [load_image(path) for path in paths]

    orig = images[0]
    display_images = [orig]
    titles = ["Original"]

    for idx, img in enumerate(images[1:], start=1):
        scaled = np.clip(orig + (img - orig) * args.times, 0.0, 1.0)
        display_images.append(scaled)
        titles.append(f"{IMAGE_FILES[idx]} x{args.times:g}")

    num_images = len(display_images)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1:
        axes = [axes]

    for ax, img, title in zip(axes, display_images, titles):
        ax.imshow(img.squeeze() if img.shape[-1] == 1 else img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    if args.output is not None:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = DEFAULT_PIC_DIR / output_path
    else:
        DEFAULT_PIC_DIR.mkdir(parents=True, exist_ok=True)
        stems = "_".join(path.stem for path in paths)
        output_name = f"{stems}.png"
        output_path = DEFAULT_PIC_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"已保存图像到 {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

