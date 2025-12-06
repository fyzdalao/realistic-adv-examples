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
DEFAULT_PIC_DIR = SCRIPT_DIR / "pics"

# ========== 可在此处直接修改配置 ==========

# 默认目录配置（如果通过命令行参数提供，则使用命令行参数）

# DEFAULT_DIR1 = "results/resnet_imagenet/linf/hsja/1205-02-01-46_discrete-0_targeted-0_early-0_binary_0.000"
# DEFAULT_DIR2 = "results/resnet_imagenet/linf/hsja/1205-03-25-47_discrete-0_targeted-0_early-0_binary_0.000"
#26 56


# DEFAULT_DIR1 = "results/resnet_imagenet/l2/hsja/1204-09-04-04_discrete-0_targeted-0_early-0_binary_0.000"
# DEFAULT_DIR2 = "results/resnet_imagenet/l2/hsja/1204-09-15-44_discrete-0_targeted-0_early-0_binary_0.000"


# DEFAULT_DIR1 = "results/resnet_imagenet/l2/geoda/1204-07-38-19_discrete-0_targeted-0_early-0_binary_0.000"
# DEFAULT_DIR2 = "results/resnet_imagenet/l2/geoda/1204-07-38-43_discrete-0_targeted-0_early-0_binary_0.000"
# 34 118


DEFAULT_DIR1 = "results/resnet_imagenet/l2/sign_opt/1205-06-15-16_discrete-0_targeted-0_early-0_binary_0.000"
DEFAULT_DIR2 = "results/resnet_imagenet/l2/sign_opt/1205-06-15-21_discrete-0_targeted-0_early-0_binary_0.000"
# #125 124 121 117 81


DEFAULT_INDEX = 81
DEFAULT_TIMES = 1.0  # 放大对抗扰动的倍数（orig 保持不变）
USE_PSD_FOR_THIRD_IMAGE = False  # True: 第三幅图使用PSD防御模型, False: 第三幅图使用无防御模型


# ==========================================


def load_image(path: Path) -> np.ndarray:
    img = np.load(path)
    if img.ndim == 3:
        if img.shape[0] in {1, 3}:
            img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = np.clip(img, 0.0, 1.0)
    return img


def load_model(device: torch.device, defense: str = 'none') -> TorchModelWrapper:
    """加载 ResNet50 模型
    
    Args:
        device: 设备
        defense: 防御类型，'none' 或 'PSD'
    """
    inner_model = models.__dict__["resnet50"](weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
    
    # 根据防御类型设置参数
    if defense == 'PSD':
        # 使用与main.py中defense=PSD相同的参数
        pawn_thres_prob = 0.1
        pawn_thres_logits = 0.2
        model = TorchModelWrapper(
            inner_model,
            n_class=1000,
            im_mean=(0.485, 0.456, 0.406),
            im_std=(0.229, 0.224, 0.225),
            defense='PSD',
            pawn_thres_prob=pawn_thres_prob,
            pawn_thres_logits=pawn_thres_logits
        )
    else:
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
    parser.add_argument("--dir1",
                        default=None,
                        type=str,
                        help=f"第一个目录路径（包含details文件夹，用于加载原图和第一个对抗样本）。如果不提供，使用代码中的默认值: {DEFAULT_DIR1}")
    parser.add_argument("--dir2",
                        default=None,
                        type=str,
                        help=f"第二个目录路径（包含details文件夹，用于加载第二个对抗样本）。如果不提供，使用代码中的默认值: {DEFAULT_DIR2}")
    parser.add_argument("--index", "-i",
                        default=None,
                        type=int,
                        help=f"图像编号（例如：117）。如果不提供，使用代码中的默认值: {DEFAULT_INDEX}")
    parser.add_argument("--output",
                        default=None,
                        help=f"输出PDF路径（默认保存到 {DEFAULT_PIC_DIR} 下并自动命名，格式为PDF）")
    parser.add_argument("--times",
                        default=None,
                        type=float,
                        help=f"放大对抗扰动的倍数（orig 保持不变）。如果不提供，使用代码中的默认值: {DEFAULT_TIMES}")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="使用的 GPU 编号（默认：0）")
    args = parser.parse_args()

    # 使用命令行参数或默认值
    dir1_str = args.dir1 if args.dir1 is not None else DEFAULT_DIR1
    dir2_str = args.dir2 if args.dir2 is not None else DEFAULT_DIR2
    index = args.index if args.index is not None else DEFAULT_INDEX
    times = args.times if args.times is not None else DEFAULT_TIMES
    use_psd_for_third = USE_PSD_FOR_THIRD_IMAGE
    
    # 解析目录路径（如果不是绝对路径，则相对于项目根目录）
    dir1 = Path(dir1_str)
    dir2 = Path(dir2_str)
    
    # 如果不是绝对路径，则相对于项目根目录
    if not dir1.is_absolute():
        dir1 = PROJECT_ROOT / dir1
    if not dir2.is_absolute():
        dir2 = PROJECT_ROOT / dir2
    
    # 检查目录是否存在
    if not dir1.exists():
        raise ValueError(f"第一个目录不存在: {dir1}")
    if not dir2.exists():
        raise ValueError(f"第二个目录不存在: {dir2}")
    
    # 构建文件路径
    details_dir1 = dir1 / "details"
    details_dir2 = dir2 / "details"
    
    if not details_dir1.exists():
        raise ValueError(f"第一个目录下不存在 details 文件夹: {details_dir1}")
    if not details_dir2.exists():
        raise ValueError(f"第二个目录下不存在 details 文件夹: {details_dir2}")
    
    # 构建图像文件路径
    orig_path = details_dir1 / f"{index}.npy"
    adv1_path = details_dir1 / f"{index}_adv.npy"
    adv2_path = details_dir2 / f"{index}_adv.npy"
    
    # 检查文件是否存在
    if not orig_path.exists():
        raise ValueError(f"原图文件不存在: {orig_path}")
    if not adv1_path.exists():
        raise ValueError(f"第一个对抗样本文件不存在: {adv1_path}")
    if not adv2_path.exists():
        raise ValueError(f"第二个对抗样本文件不存在: {adv2_path}")
    
    print(f"加载图像:")
    print(f"  原图: {orig_path}")
    print(f"  第一个对抗样本: {adv1_path}")
    print(f"  第二个对抗样本: {adv2_path}")

    # 设置设备
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，无法使用 GPU。")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"使用 GPU {args.gpu}")

    # 加载模型
    print("正在加载 ResNet50 模型（无防御）...")
    model_no_defense = load_model(device, defense='none')
    print("无防御模型加载完成")
    
    print("正在加载 ResNet50 模型（PSD防御）...")
    model_psd = load_model(device, defense='PSD')
    print("PSD防御模型加载完成")

    # 加载图像
    orig = load_image(orig_path)
    adv1 = load_image(adv1_path)
    adv2 = load_image(adv2_path)
    images = [orig, adv1, adv2]

    display_images = [orig]
    
    # 定义三个图的标题
    image_titles = ["原始图像", "无防御的攻击结果", "防御下攻击结果"]
    titles = []

    # 对原图进行预测（使用无防御模型）
    orig_class_idx, orig_class_name = predict_image(model_no_defense, orig, device)
    titles.append(f"{image_titles[0]}\n预测类别: {orig_class_name} ({orig_class_idx})")

    for idx, img in enumerate(images[1:], start=1):
        scaled = np.clip(orig + (img - orig) * times, 0.0, 1.0)
        display_images.append(scaled)
        
        # 对前两个图像（索引1）使用无防御模型，对第三个图像（索引2）根据配置选择模型
        if idx == 1:
            # 第二个图像：使用无防御模型
            pred_class_idx, pred_class_name = predict_image(model_no_defense, scaled, device)
        else:
            # 第三个图像：根据配置选择使用PSD防御模型或无防御模型
            if use_psd_for_third:
                pred_class_idx, pred_class_name = predict_image(model_psd, scaled, device)
            else:
                pred_class_idx, pred_class_name = predict_image(model_no_defense, scaled, device)
        
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
        # 从路径中提取攻击方法和范数类型
        # 路径格式: results/resnet_imagenet/{norm}/{attack}/{timestamp_...}
        parts1 = dir1.parts
        parts2 = dir2.parts
        
        # 查找 norm 和 attack（通常在路径中包含 'l2', 'linf', 'hsja', 'rays' 等）
        norm = None
        attack = None
        
        # 从路径中提取（假设路径结构为 .../norm/attack/...）
        for i, part in enumerate(parts1):
            if part in ['l2', 'linf']:
                norm = part
            elif part in ['hsja', 'rays', 'signopt', 'geoda', 'opt', 'boundary']:
                attack = part
        
        # 如果没找到，尝试从 dir2 找
        if not norm or not attack:
            for i, part in enumerate(parts2):
                if part in ['l2', 'linf'] and not norm:
                    norm = part
                elif part in ['hsja', 'rays', 'signopt', 'geoda', 'opt', 'boundary'] and not attack:
                    attack = part
        
        # 如果还是没找到，使用默认值
        if not norm:
            norm = "unknown"
        if not attack:
            attack = "unknown"
        
        # 生成简化的输出文件名: {attack}_{norm}_{index}_times{times}.pdf
        # 将 times 格式化为整数或保留一位小数
        if times == int(times):
            times_str = f"{int(times)}"
        else:
            times_str = f"{times:.1f}".replace('.', 'p')  # 将小数点替换为p，避免文件名问题
        output_name = f"{attack}_{norm}_{index}_times{times_str}.pdf"
        output_path = DEFAULT_PIC_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 保存为PDF格式，文字将以矢量形式保存，可以无损缩放
    plt.savefig(output_path, bbox_inches="tight", format='pdf', dpi=300)
    print(f"已保存图像到 {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()



