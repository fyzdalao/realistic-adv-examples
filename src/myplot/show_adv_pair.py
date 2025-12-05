#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib
# 配置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import numpy as np
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights

from src.model_wrappers import TorchModelWrapper
DEFAULT_NPY_DIR = SCRIPT_DIR / "npys"
DEFAULT_PIC_DIR = SCRIPT_DIR / "pics"
IMAGE_FILES = [
    "34.npy",
    "34_adv.npy",
    "34_advv.npy",
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


def load_model(device: torch.device) -> TorchModelWrapper:
    """加载 ResNet50 模型"""
    inner_model = models.__dict__["resnet50"](weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
    model = TorchModelWrapper(
        inner_model,
        n_class=1000,
        im_mean=(0.485, 0.456, 0.406),
        im_std=(0.229, 0.224, 0.225),
        defense='none'
    )
    model.make_model_eval()
    return model


# 全局变量缓存类别名称
_IMAGENET_CLASSES: dict[int, str] | None = None


def _load_imagenet_classes() -> dict[int, str]:
    """加载 ImageNet 类别名称映射"""
    # 首先尝试从本地文件加载
    classes_file = SCRIPT_DIR / "imagenet_classes.txt"
    if classes_file.exists():
        try:
            with open(classes_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
                if len(classes) >= 1000:
                    return {i: classes[i] for i in range(1000)}
        except Exception as e:
            print(f"无法从本地文件加载 ImageNet 类别名称: {e}")
    
    # 尝试从网上下载 ImageNet 类别名称
    try:
        import urllib.request
        
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        print("正在从网络加载 ImageNet 类别名称...")
        with urllib.request.urlopen(url, timeout=10) as response:
            classes = [line.decode('utf-8').strip() for line in response.readlines() if line.strip()]
            if len(classes) >= 1000:
                print(f"成功加载 {len(classes)} 个类别名称")
                # 同时保存到本地文件以便下次使用
                try:
                    with open(classes_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(classes))
                except:
                    pass
                return {i: classes[i] for i in range(1000)}
    except Exception as e:
        print(f"无法从网络加载 ImageNet 类别名称: {e}")
        print("提示：可以手动下载 imagenet_classes.txt 文件到脚本目录")
    
    # 如果都失败，返回空字典
    return {}


def get_imagenet_class_name(class_idx: int) -> str:
    """获取 ImageNet 类别名称"""
    global _IMAGENET_CLASSES
    
    if _IMAGENET_CLASSES is None:
        _IMAGENET_CLASSES = _load_imagenet_classes()
    
    if isinstance(_IMAGENET_CLASSES, dict) and class_idx in _IMAGENET_CLASSES:
        return _IMAGENET_CLASSES[class_idx]
    
    # 如果无法获取类别名称，返回类别索引
    return f"Class {class_idx}"


def predict_image(model: TorchModelWrapper, img: np.ndarray, device: torch.device) -> tuple[int, str]:
    """对图像进行预测，返回类别索引和名称"""
    # 将图像转换为 torch tensor
    # 图像格式应该是 (H, W, C)，需要转换为 (C, H, W)
    if img.ndim == 3:
        if img.shape[2] == 3:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        elif img.shape[2] == 1:
            img_tensor = torch.from_numpy(img).squeeze(-1).float()
            img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    else:
        img_tensor = torch.from_numpy(img).float()
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
    
    # 确保值在 [0, 1] 范围内
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    
    # 添加 batch 维度并移动到设备
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 进行预测
    with torch.no_grad():
        pred_label = model.predict_label(img_tensor)
        class_idx = int(pred_label.item())
    
    # 获取类别名称（简化版本，实际可以使用 ImageNet 的类别映射）
    class_name = get_imagenet_class_name(class_idx)
    
    return class_idx, class_name


def main():
    parser = argparse.ArgumentParser(description="显示原图与对抗样本")
    parser.add_argument("--output",
                        default=None,
                        help=f"输出PDF路径（默认保存到 {DEFAULT_PIC_DIR} 下并自动命名，格式为PDF）")
    parser.add_argument("--times",
                        type=float,
                        default=1.0,
                        help="放大对抗扰动的倍数（orig 保持不变）")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="使用的 GPU 编号（默认：1）")
    args = parser.parse_args()

    if not IMAGE_FILES:
        raise ValueError("IMAGE_FILES 列表为空，请在脚本中配置需要展示的 npy 路径。")

    # 设置设备
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，无法使用 GPU。")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"使用 GPU {args.gpu}")

    # 加载模型
    print("正在加载 ResNet50 模型...")
    model = load_model(device)
    print("模型加载完成")

    paths = [resolve_path(name) for name in IMAGE_FILES]
    images = [load_image(path) for path in paths]

    orig = images[0]
    display_images = [orig]
    
    # 定义三个图的标题
    image_titles = ["原始图像", "无防御的攻击结果", "防御下攻击结果"]
    titles = []

    # 对原图进行预测
    orig_class_idx, orig_class_name = predict_image(model, orig, device)
    titles.append(f"{image_titles[0]}\n预测类别: {orig_class_name} ({orig_class_idx})")

    for idx, img in enumerate(images[1:], start=1):
        scaled = np.clip(orig + (img - orig) * args.times, 0.0, 1.0)
        display_images.append(scaled)
        
        # 对缩放后的图像进行预测
        pred_class_idx, pred_class_name = predict_image(model, scaled, device)
        # 使用对应的中文标题，不显示文件名
        title_idx = min(idx, len(image_titles) - 1)
        titles.append(f"{image_titles[title_idx]}\n预测类别: {pred_class_name} ({pred_class_idx})")

    num_images = len(display_images)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1:
        axes = [axes]

    for ax, img, title in zip(axes, display_images, titles):
        ax.imshow(img.squeeze() if img.shape[-1] == 1 else img)
        ax.set_title(title, fontsize=22)
        ax.axis("off")

    plt.tight_layout()

    if args.output is not None:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = DEFAULT_PIC_DIR / output_path
        # 如果没有扩展名，添加.pdf
        if not output_path.suffix:
            output_path = output_path.with_suffix('.pdf')
    else:
        DEFAULT_PIC_DIR.mkdir(parents=True, exist_ok=True)
        stems = "_".join(path.stem for path in paths)
        output_name = f"{stems}.pdf"
        output_path = DEFAULT_PIC_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 保存为PDF格式，文字将以矢量形式保存，可以无损缩放
    plt.savefig(output_path, bbox_inches="tight", format='pdf', dpi=300)
    print(f"已保存图像到 {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

