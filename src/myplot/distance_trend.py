
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Locator
from collections import defaultdict

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class ThresholdLocator(Locator):
    """
    自定义Locator，在阈值上下分别平均分割刻度
    """
    def __init__(self, threshold_value, num_ticks_below=5, num_ticks_above=5):
        """
        Args:
            threshold_value: 阈值
            num_ticks_below: 阈值以下的刻度数量
            num_ticks_above: 阈值以上的刻度数量
        """
        self.threshold_value = threshold_value
        self.num_ticks_below = num_ticks_below
        self.num_ticks_above = num_ticks_above
    
    def __call__(self):
        if self.axis is None:
            return []
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)
    
    def tick_values(self, vmin, vmax):
        # 阈值以下的刻度
        if vmin < self.threshold_value:
            ticks_below = np.linspace(vmin, self.threshold_value, self.num_ticks_below + 1)
            ticks_below = ticks_below[:-1].tolist()  # 排除阈值本身，避免重复，转换为列表
        else:
            ticks_below = []
        
        # 阈值以上的刻度
        if vmax > self.threshold_value:
            ticks_above = np.linspace(self.threshold_value, vmax, self.num_ticks_above + 1)
            ticks_above = ticks_above.tolist()
        else:
            ticks_above = []
        
        # 合并刻度，包括阈值本身
        if len(ticks_below) > 0 and len(ticks_above) > 0:
            ticks = ticks_below + [self.threshold_value] + ticks_above
        elif len(ticks_below) > 0:
            ticks = ticks_below + [self.threshold_value]
        elif len(ticks_above) > 0:
            ticks = [self.threshold_value] + ticks_above
        else:
            ticks = [self.threshold_value]
        
        return ticks


def load_distance_data(details_dir, allowed_files=None):
    """
    读取details目录下所有.dis文件的数据
    
    Args:
        details_dir: details文件夹的路径
        allowed_files: 可选，允许处理的文件名集合（只包含文件名，不包含路径）
        
    Returns:
        dict: 键为迭代步数，值为该步数对应的所有距离值的列表
    """
    # 存储每个迭代步数对应的所有距离值
    step_distances = defaultdict(list)
    
    # 查找所有.dis文件
    dis_files = glob.glob(os.path.join(details_dir, "*.dis"))
    
    # 如果指定了允许的文件列表，则只处理这些文件
    if allowed_files is not None:
        dis_files = [f for f in dis_files if os.path.basename(f) in allowed_files]
    
    print(f"找到 {len(dis_files)} 个.dis文件")
    
    # 读取每个文件
    for dis_file in dis_files:
        try:
            max_step = -1
            last_distance = None
            
            with open(dis_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            step = int(parts[0])
                            distance = float(parts[1])
                            step_distances[step].append(distance)
                            
                            # 记录该文件的最大迭代步数和最后一次的距离值
                            if step > max_step:
                                max_step = step
                                last_distance = distance
            
            # 如果该文件的迭代次数不到50000，用最后一次迭代的距离值填充
            if max_step < 50000 and last_distance is not None:
                for fill_step in range(max_step + 1, 50001):
                    step_distances[fill_step].append(last_distance)
                    
        except Exception as e:
            print(f"读取文件 {dis_file} 时出错: {e}")
            continue
    
    return step_distances

class CompressedScale(mscale.ScaleBase):
    """
    自定义y轴缩放，压缩早期大值占用的空间
    """
    name = 'compressed'
    
    def __init__(self, axis, *, compression_ratio=0.3, threshold_value=None, **kwargs):
        """
        Args:
            compression_ratio: 压缩比例，大值区域占用的空间比例
            threshold_value: 阈值，超过此值的数据会被压缩显示
        """
        super().__init__(axis)
        self.compression_ratio = compression_ratio
        self.threshold_value = threshold_value
    
    def get_transform(self):
        return self.CompressedTransform(self.compression_ratio, self.threshold_value)
    
    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(plt.AutoLocator())
        axis.set_major_formatter(plt.ScalarFormatter())
        axis.set_minor_formatter(plt.NullFormatter())
    
    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax
    
    class CompressedTransform(mtransforms.Transform):
        def __init__(self, compression_ratio, threshold_value):
            super().__init__()
            self.compression_ratio = compression_ratio
            self.threshold_value = threshold_value
        
        @property
        def input_dims(self):
            return 1
        
        @property
        def output_dims(self):
            return 1
        
        def transform_non_affine(self, values):
            values = np.asarray(values)
            if self.threshold_value is None:
                return values
            
            # 分段线性变换
            result = np.zeros_like(values)
            mask_low = values <= self.threshold_value
            mask_high = values > self.threshold_value
            
            # 低值区域：正常显示
            result[mask_low] = values[mask_low]
            
            # 高值区域：压缩显示
            if np.any(mask_high):
                excess = values[mask_high] - self.threshold_value
                result[mask_high] = self.threshold_value + excess * self.compression_ratio
            
            return result
        
        def inverted(self):
            return CompressedScale.InvertedCompressedTransform(self.compression_ratio, self.threshold_value)
    
    class InvertedCompressedTransform(mtransforms.Transform):
        def __init__(self, compression_ratio, threshold_value):
            super().__init__()
            self.compression_ratio = compression_ratio
            self.threshold_value = threshold_value
        
        @property
        def input_dims(self):
            return 1
        
        @property
        def output_dims(self):
            return 1
        
        def transform_non_affine(self, values):
            values = np.asarray(values)
            if self.threshold_value is None:
                return values
            
            result = np.zeros_like(values)
            mask_low = values <= self.threshold_value
            mask_high = values > self.threshold_value
            
            # 低值区域：正常显示
            result[mask_low] = values[mask_low]
            
            # 高值区域：解压缩
            if np.any(mask_high):
                excess = values[mask_high] - self.threshold_value
                result[mask_high] = self.threshold_value + excess / self.compression_ratio
            
            return result

# 注册自定义scale
mscale.register_scale(CompressedScale)

def smooth_curve(data, steps, window_size=5, method='median'):
    """
    平滑曲线，去除突然的尖锐峰值
    
    Args:
        data: 要平滑的数据数组（一维数组）
        steps: 对应的迭代步数数组（用于判断是否需要过滤）
        window_size: 平滑窗口大小（必须是奇数）
        method: 平滑方法，'median' 使用中位数滤波，'moving_average' 使用移动平均
        
    Returns:
        平滑后的数据数组
    """
    if len(data) < window_size:
        return data
    
    if len(data) != len(steps):
        raise ValueError("data和steps的长度必须相同")
    
    # 确保窗口大小是奇数
    if window_size % 2 == 0:
        window_size += 1
    
    data_array = np.array(data)
    steps_array = np.array(steps)
    smoothed = np.copy(data_array)
    half_window = window_size // 2
    
    if method == 'median':
        # 使用中位数滤波，对去除突然的峰值很有效
        for i in range(len(data_array)):
            # 如果横轴坐标小于10，不进行平滑处理
            if steps_array[i] < 10:
                smoothed[i] = data_array[i]
                continue
            
            # 计算窗口范围
            start = max(0, i - half_window)
            end = min(len(data_array), i + half_window + 1)
            
            # 确保窗口内的数据点足够多
            window_data = data_array[start:end]
            if len(window_data) >= 3:  # 至少需要3个点才能计算中位数
                smoothed[i] = np.median(window_data)
            else:
                smoothed[i] = data_array[i]
        
        return smoothed
    else:
        # 使用移动平均
        for i in range(len(data_array)):
            # 如果横轴坐标小于10，不进行平滑处理
            if steps_array[i] < 10:
                smoothed[i] = data_array[i]
                continue
            
            # 计算窗口范围
            start = max(0, i - half_window)
            end = min(len(data_array), i + half_window + 1)
            
            # 计算移动平均
            window_data = data_array[start:end]
            if len(window_data) > 0:
                smoothed[i] = np.mean(window_data)
            else:
                smoothed[i] = data_array[i]
        
        return smoothed

def calculate_statistics(step_distances):
    """
    计算每个迭代步数的平均值和中位数
    
    Args:
        step_distances: 从load_distance_data返回的字典
        
    Returns:
        tuple: (迭代步数列表, 平均值列表, 中位数列表)
    """
    # 获取所有迭代步数并排序
    steps = sorted(step_distances.keys())
    
    means = []
    medians = []
    
    for step in steps:
        distances = step_distances[step]
        
        if len(distances) > 0:
            means.append(np.mean(distances))
            medians.append(np.median(distances))
        else:
            # 如果没有有效数据，使用NaN
            means.append(np.nan)
            medians.append(np.nan)
    
    return steps, means, medians

def plot_two_logs_comparison(log_a_dir, log_b_dir, threshold_value, y_min, y_max, y_ticks):
    """
    绘制两个日志的距离趋势对比图
    
    Args:
        log_a_dir: 日志A的目录路径（包含details文件夹的目录）
        log_b_dir: 日志B的目录路径（包含details文件夹的目录）
        threshold_value: y轴压缩阈值
        y_min: y轴最小值
        y_max: y轴最大值
        y_ticks: y轴刻度点列表
    """
    # 获取两个日志的details目录
    details_a_dir = os.path.join(log_a_dir, "details")
    if not os.path.exists(details_a_dir):
        raise ValueError(f"找不到日志A的details目录: {details_a_dir}")
    
    details_b_dir = os.path.join(log_b_dir, "details")
    if not os.path.exists(details_b_dir):
        raise ValueError(f"找不到日志B的details目录: {details_b_dir}")
    
    # 找出两个日志中共同存在的.dis文件名
    dis_files_a = set(os.path.basename(f) for f in glob.glob(os.path.join(details_a_dir, "*.dis")))
    dis_files_b = set(os.path.basename(f) for f in glob.glob(os.path.join(details_b_dir, "*.dis")))
    common_files = dis_files_a & dis_files_b  # 交集
    
    print(f"日志A中有 {len(dis_files_a)} 个.dis文件")
    print(f"日志B中有 {len(dis_files_b)} 个.dis文件")
    print(f"共同存在的文件有 {len(common_files)} 个")
    
    if len(common_files) == 0:
        raise ValueError("两个日志中没有共同存在的.dis文件")
    
    # 只处理共同存在的文件
    print("处理日志A...")
    step_distances_a = load_distance_data(details_a_dir, allowed_files=common_files)
    if not step_distances_a:
        raise ValueError("日志A没有找到任何有效的数据")
    steps_a, means_a, medians_a = calculate_statistics(step_distances_a)
    
    print("处理日志B...")
    step_distances_b = load_distance_data(details_b_dir, allowed_files=common_files)
    if not step_distances_b:
        raise ValueError("日志B没有找到任何有效的数据")
    steps_b, means_b, medians_b = calculate_statistics(step_distances_b)
    
    # 过滤掉横轴坐标大于50000的数据点
    max_step = 50000
    mask_a = np.array(steps_a) <= max_step
    mask_b = np.array(steps_b) <= max_step
    
    steps_a_filtered = np.array(steps_a)[mask_a]
    means_a_filtered = np.array(means_a)[mask_a]
    medians_a_filtered = np.array(medians_a)[mask_a]
    
    steps_b_filtered = np.array(steps_b)[mask_b]
    means_b_filtered = np.array(means_b)[mask_b]
    medians_b_filtered = np.array(medians_b)[mask_b]
    
    # 对最终的曲线进行平滑处理，去除突然的峰值
    window_size = 11  # 平滑窗口大小
    means_a_smooth = smooth_curve(means_a_filtered, steps_a_filtered, window_size=window_size, method='median')
    medians_a_smooth = smooth_curve(medians_a_filtered, steps_a_filtered, window_size=window_size, method='median')
    means_b_smooth = smooth_curve(means_b_filtered, steps_b_filtered, window_size=window_size, method='median')
    medians_b_smooth = smooth_curve(medians_b_filtered, steps_b_filtered, window_size=window_size, method='median')
    
    # 创建单个图：只显示平滑处理后的数据（增大图像尺寸以确保清晰度）
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # 设置y轴使用自定义的压缩scale
    ax.set_yscale('compressed', compression_ratio=0.1, threshold_value=threshold_value)
    
    # 绘制平滑处理后的数据（不进行数据压缩，让scale处理）
    ax.plot(steps_b_filtered, means_b_smooth, label='防御 - 平均', linewidth=2.5, color='#ff7f0e', linestyle='-')
    ax.plot(steps_b_filtered, medians_b_smooth, label='防御 - 中位', linewidth=2.5, color='#ff7f0e', linestyle=':', alpha=0.8)
    ax.plot(steps_a_filtered, means_a_smooth, label='无防御-平均', linewidth=2.5, color='#1f77b4', linestyle='-')
    ax.plot(steps_a_filtered, medians_a_smooth, label='无防御-中位', linewidth=2.5, color='#1f77b4', linestyle=':', alpha=0.8)

    
    # 设置横轴范围，最小接近0，最大接近50000，以节约空间
    all_steps = np.concatenate([steps_a_filtered, steps_b_filtered])
    x_min =  -1000  # 最小值不要小于0太多，最多减去50
    x_max = min(50000, np.max(all_steps)) + 1000  # 最大值不要超出50000太多，最多加上50
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # 设置y轴刻度
    ax.set_yticks(y_ticks)
    
    # 增大所有文字字号，确保在毕业论文中清晰可见
    ax.set_xlabel('模型访问次数', fontsize=28, fontweight='bold')
    ax.set_ylabel('距离', fontsize=28, fontweight='bold')
    # ax.set_title('距离随迭代步数的变化趋势', fontsize=32, fontweight='bold', pad=20)
    ax.legend(fontsize=26, loc='best', framealpha=0.9)
    
    # 增大坐标轴刻度标签字号
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    
    ax.grid(True, alpha=0.3, linewidth=1.5)
    
    plt.tight_layout()
    
    # 保存为PDF格式到myplot文件夹
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "distance_trend.pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"图像已保存为PDF: {pdf_path}")
    
    # 显示图片
    plt.show()

if __name__ == "__main__":
    # ========== 配置参数 ==========
    # 指定两个日志目录


    # geoda
    log_a_dir = r"results\resnet_imagenet\l2\geoda\1204-07-38-19_discrete-0_targeted-0_early-0_binary_0.000"
    log_b_dir = r"results\resnet_imagenet\l2\geoda\1204-07-38-43_discrete-0_targeted-0_early-0_binary_0.000"
    threshold_value = 50
    y_min = -1
    y_max = 160
    y_ticks = [0, 10, 20, 30, 40, 50, 70, 90, 110, 130, 150]
    
    # # hsja l2
    # log_a_dir = r"results\resnet_imagenet\l2\hsja\1204-09-04-04_discrete-0_targeted-0_early-0_binary_0.000"
    # log_b_dir = r"results\resnet_imagenet\l2\hsja\1204-09-15-44_discrete-0_targeted-0_early-0_binary_0.000"
    # threshold_value = 40
    # y_min = -1
    # y_max = 160
    # y_ticks = [0, 10, 20, 30, 40, 60, 80, 100, 120, 140]

    # # hsja linf
    # log_a_dir = r"results\resnet_imagenet\linf\hsja\1205-02-01-46_discrete-0_targeted-0_early-0_binary_0.000"
    # log_b_dir = r"results\resnet_imagenet\linf\hsja\1205-03-25-47_discrete-0_targeted-0_early-0_binary_0.000"
    # threshold_value = 0.3
    # y_min = 0
    # y_max = 1
    # y_ticks = [0, 0.10, 0.20, 0.30, 0.50, 0.7, 0.9]

    # signOPT
    # log_a_dir = r"results\resnet_imagenet\l2\sign_opt\1205-06-15-16_discrete-0_targeted-0_early-0_binary_0.000"
    # log_b_dir = r"results\resnet_imagenet\l2\sign_opt\1205-06-15-21_discrete-0_targeted-0_early-0_binary_0.000"
    # threshold_value = 50
    # y_min = -1
    # y_max = 160
    # y_ticks = [0, 10, 20, 30, 40, 50, 70, 90, 110, 130, 150]


    # ============================
    
    # 检查目录是否存在
    if not os.path.exists(log_a_dir):
        print(f"警告: 目录 {log_a_dir} 不存在")
        print("请修改log_a_dir变量为正确的路径")
    elif not os.path.exists(log_b_dir):
        print(f"警告: 目录 {log_b_dir} 不存在")
        print("请修改log_b_dir变量为正确的路径")
    else:
        plot_two_logs_comparison(log_a_dir, log_b_dir, threshold_value, y_min, y_max, y_ticks)


