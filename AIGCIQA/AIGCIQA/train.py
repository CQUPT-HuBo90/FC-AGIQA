import os
import sys

# 首先从命令行获取GPU参数并设置环境变量（必须在import torch之前）
def get_gpu_from_args():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpu', type=str, default='0', help='id of gpu device(s) to be used')
    args, _ = parser.parse_known_args()
    return args.gpu

# 立即设置GPU环境变量
gpu_id = get_gpu_from_args()
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
print(f"🔧 [启动时] 设置CUDA_VISIBLE_DEVICES={gpu_id}")

import torch

# 限制每个训练进程的CPU线程数，避免多进程训练时CPU线程过度订阅导致GPU等待数据
_TORCH_NUM_THREADS = int(os.environ.get('TORCH_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '2')))
torch.set_num_threads(_TORCH_NUM_THREADS)
torch.set_num_interop_threads(int(os.environ.get('TORCH_INTEROP_NUM_THREADS', '1')))

import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.stats import kendalltau
from tqdm import tqdm

# 导入CLIP-ConvNeXt编码器
try:
    from backbone.clip_convnext_encoder import get_clip_convnext_encoder
    CLIPCONVNEXT_AVAILABLE = True
except ImportError:
    CLIPCONVNEXT_AVAILABLE = False
    print("警告: 未找到CLIP-ConvNeXt编码器，请确保已安装open_clip_torch")
# 注释掉常规数据加载器导入，只使用交叉验证模式
# from dataloader.AGIQA1K import get_AGIQA1K_dataloaders
# from dataloader.AGIQA3K import get_AGIQA3K_dataloaders
# from dataloader.AIGCIQA2023 import get_AIGCIQA2023q_dataloaders, get_AIGCIQA2023a_dataloaders, get_AIGCIQA2023c_dataloaders

# 导入交叉验证数据加载器
from dataloader.AGIQA1K_CV import get_AGIQA1K_CV_dataloaders
from dataloader.AGIQA3K_CV import get_AGIQA3K_CV_dataloaders, get_AGIQA3Kc_CV_dataloaders
from dataloader.AIGCIQA2023_CV import get_AIGCIQA2023_CV_dataloaders
from dataloader.AIGCIQA20K import get_AIGCIQA20K_dataloaders
from dataloader.I2IQA_CV import get_I2IQA_CV_dataloaders
from dataloader.cv_utils import create_cv_dataloader, get_image_transforms
from model import Encoder, MLP
from config import get_parser
from util import get_logger, log_and_print
from datetime import datetime
import csv
import fcntl

# 从环境变量读取DEBUG_MODE，与model.py保持一致
DEBUG_MODE = os.environ.get("DEBUG_MODE", "true").lower() == "true"


# 添加必要的包依赖
try:
    import kornia.color
except ImportError:
    print("警告: 未找到kornia包，请使用以下命令安装:")
    print("pip install kornia")

import random
import numpy as np  # 添加numpy导入以支持完整的随机种子设置
import torch.backends.cudnn as cudnn
import math

# 导入高级训练策略
try:
    from utils.advanced_training_strategies import (
        CosineAnnealingWarmRestarts,
        MixupAugmentation, 
        ProgressiveLearning,
        AdaptiveLossWeighting,
        EMA,
        GradientAccumulation,
        get_advanced_optimizer,
        DataBalancedSampler
    )
    ADVANCED_TRAINING_AVAILABLE = True
except ImportError:
    ADVANCED_TRAINING_AVAILABLE = False
    print("警告: 未找到高级训练策略模块，将使用标准训练")

sys.path.append('../')


def get_swin_extractor():
    """按需导入Swin，避免clipconvnext实验触发timm/wandb依赖链。"""
    from backbone.vit import SwinExtractor
    return SwinExtractor

# 数据加载器工作进程随机种子初始化函数
def worker_init_fn(worker_id):
    """
    数据加载器工作进程随机种子初始化函数
    确保每个worker使用不同但确定的随机种子
    """
    # 从环境变量获取全局种子，如果没有则使用默认值
    global_seed = int(os.environ.get('GLOBAL_SEED', 42))
    worker_seed = global_seed + worker_id
    
    # 设置该worker的随机种子
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    # 如果使用CUDA，设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)

# 完整的确定性设置函数
def setup_deterministic_training(seed):
    """
    设置完全确定性的训练环境
    统一使用来自run脚本的种子值
    """
    print(f"🔧 设置完全确定性训练环境，种子值: {seed}")
    
    # 将种子保存到环境变量，供worker_init_fn使用
    os.environ['GLOBAL_SEED'] = str(seed)
    
    # 设置Python随机种子
    random.seed(seed)
    
    # 设置NumPy随机种子
    np.random.seed(seed)
    
    # 设置PyTorch随机种子
    torch.manual_seed(seed)
    
    # 设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子
    
    # 设置CuDNN为确定性模式（禁用benchmark以确保确定性）
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False  # 禁用benchmark以确保完全确定性
    torch.backends.cudnn.deterministic = True  # 启用确定性模式
    
    # 设置PyTorch的其他确定性行为
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print("✅ 确定性训练环境设置完成")
    print(f"   - Python random seed: {seed}")
    print(f"   - NumPy random seed: {seed}")
    print(f"   - PyTorch manual seed: {seed}")
    print(f"   - CUDA seeds: {seed} (所有GPU)")
    print(f"   - CuDNN deterministic: True")
    print(f"   - CuDNN benchmark: False (为确保确定性)")
    print(f"   - PyTorch deterministic algorithms: True")

def compute_global_contrastive_loss(image_features, text_features, temperature=0.07):
    """
    批内InfoNCE对比损失（图像-文本双向）
    仅用于一致性任务的辅助对齐
    """
    if image_features.size(0) < 2:
        return image_features.new_tensor(0.0)

    temperature = max(float(temperature), 1e-6)
    image_norm = F.normalize(image_features, p=2, dim=1)
    text_norm = F.normalize(text_features, p=2, dim=1)

    logits = image_norm @ text_norm.t()
    logits = logits / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2


def compute_pearson_loss(preds, targets, eps=1e-8):
    """1 - Pearson相关系数，用于直接对齐PLCC目标。"""
    if preds.numel() < 2:
        return preds.new_tensor(0.0)

    pred_centered = preds - preds.mean()
    target_centered = targets - targets.mean()
    pred_std = torch.sqrt(torch.mean(pred_centered ** 2) + eps)
    target_std = torch.sqrt(torch.mean(target_centered ** 2) + eps)
    corr = torch.mean(pred_centered * target_centered) / (pred_std * target_std)
    return 1.0 - torch.clamp(corr, -1.0, 1.0)


def compute_pairwise_rank_loss(preds, targets, min_diff=0.03, temperature=0.5):
    """batch内成对排序损失，直接强化SRCC/KRCC。"""
    if preds.numel() < 2:
        return preds.new_tensor(0.0)

    target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
    pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
    pair_mask = torch.abs(target_diff) > min_diff
    if not pair_mask.any():
        return preds.new_tensor(0.0)

    expected_order = torch.sign(target_diff[pair_mask])
    ordered_pred_diff = pred_diff[pair_mask] * expected_order
    pair_weight = torch.abs(target_diff[pair_mask]).detach()
    pair_weight = pair_weight / (pair_weight.mean() + 1e-8)
    rank_loss = F.softplus(-ordered_pred_diff / temperature) * pair_weight
    return rank_loss.mean()


def compute_soft_weighted_rank_loss(preds, targets, temperature=0.5, eps=1e-8):
    """差值加权的成对排序损失，不使用硬阈值丢弃中段样本对。"""
    if preds.numel() < 2:
        return preds.new_tensor(0.0)

    target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
    pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
    pair_mask = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
    if not pair_mask.any():
        return preds.new_tensor(0.0)

    target_pair_diff = target_diff[pair_mask]
    pred_pair_diff = pred_diff[pair_mask]
    pair_weight = torch.abs(target_pair_diff).detach()
    nonzero_mask = pair_weight > eps
    if not nonzero_mask.any():
        return preds.new_tensor(0.0)

    expected_order = torch.sign(target_pair_diff[nonzero_mask])
    ordered_pred_diff = pred_pair_diff[nonzero_mask] * expected_order
    pair_weight = pair_weight[nonzero_mask]
    pair_weight = pair_weight / (pair_weight.mean() + eps)
    rank_loss = F.softplus(-ordered_pred_diff / max(float(temperature), eps)) * pair_weight
    return rank_loss.mean()


def compute_rank_pair_ratio(targets, min_diff=0.03):
    """统计当前batch中hard rank有效样本对比例，仅用于诊断。"""
    if targets.numel() < 2:
        return 0.0
    target_diff = torch.abs(targets.unsqueeze(1) - targets.unsqueeze(0))
    pair_mask = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
    if not pair_mask.any():
        return 0.0
    return (target_diff[pair_mask] > min_diff).float().mean().item()


def compute_quality_loss(
    preds,
    targets,
    smooth_l1_weight=0.55,
    pearson_weight=0.20,
    rank_weight=0.25,
    rank_min_diff=0.03,
    rank_temperature=0.5,
    rank_loss_type='hard',
    smooth_l1_beta=0.5
):
    """SRCC优先组合损失：稳定回归 + 线性相关 + batch内排序。"""
    smooth_l1 = F.smooth_l1_loss(preds, targets, beta=smooth_l1_beta, reduction='mean')
    pearson = compute_pearson_loss(preds, targets)
    if rank_loss_type == 'soft_weighted':
        rank = compute_soft_weighted_rank_loss(preds, targets, temperature=rank_temperature)
    else:
        rank = compute_pairwise_rank_loss(
            preds,
            targets,
            min_diff=rank_min_diff,
            temperature=rank_temperature
        )
    total = smooth_l1_weight * smooth_l1 + pearson_weight * pearson + rank_weight * rank
    return total, smooth_l1, pearson, rank


class ModelEMA:
    """训练权重的指数滑动平均，用于验证和保存更稳定的模型。"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in self._named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def _named_parameters(self):
        module = self.model.module if hasattr(self.model, 'module') else self.model
        return module.named_parameters()

    def update(self):
        with torch.no_grad():
            for name, param in self._named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self):
        self.backup = {}
        with torch.no_grad():
            for name, param in self._named_parameters():
                if name in self.shadow:
                    self.backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name])

    def restore(self):
        with torch.no_grad():
            for name, param in self._named_parameters():
                if name in self.backup:
                    param.copy_(self.backup[name])
        self.backup = {}

# 强制使用CUDA，如果不可用则退出
if not torch.cuda.is_available():
    raise RuntimeError("CUDA不可用，请确保CUDA环境正确安装和配置")

# device和avgpool将在main函数中定义，确保在设置GPU环境变量之后
avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))


# 在__main__部分之后，添加训练策略优化函数
def get_optimizer_and_scheduler(encoder, regressor, args, use_multi_gpu=False):
    """根据不同的任务和模型部分设置不同的学习率"""
    
    # 处理DataParallel包装的模型
    if use_multi_gpu:
        encoder_module = encoder.module if hasattr(encoder, 'module') else encoder
        regressor_module = regressor.module if hasattr(regressor, 'module') else regressor
    else:
        encoder_module = encoder
        regressor_module = regressor
    
    param_groups = [
        {
            'params': [p for p in encoder_module.text_encoder.parameters() if p.requires_grad],
            'lr': args.lr * args.text_lr_factor,
            'weight_decay': args.weight_decay,
            'name': 'text_encoder'
        },
        {
            'params': [p for p in encoder_module.image_encoder.parameters() if p.requires_grad],
            'lr': args.lr * args.image_lr_factor,
            'weight_decay': args.weight_decay,
            'name': 'image_encoder'
        },
        {
            'params': [p for n, p in encoder_module.named_parameters()
                       if p.requires_grad and not any(m in n for m in ['text_encoder', 'image_encoder'])],
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'name': 'fusion_encoder'
        },
        {
            'params': [p for p in regressor_module.parameters() if p.requires_grad],
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'name': 'regressor'
        }
    ]
    param_groups = [group for group in param_groups if group['params']]
    for group in param_groups:
        group['num_params'] = sum(p.numel() for p in group['params'])

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    return optimizer


def build_scheduler(optimizer, args):
    """构建epoch级学习率调度器。"""
    t_max = args.t_max if args.t_max is not None else args.num_epochs
    if args.scheduler == 'cosine' and t_max < args.num_epochs:
        print(f"⚠️  cosine T_max={t_max} 小于 num_epochs={args.num_epochs}，将使用 num_epochs 防止学习率后期回升")
        t_max = args.num_epochs
    if args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=args.min_lr
        )

    if args.scheduler == 'warmup_cosine':
        warmup_epochs = max(0, int(getattr(args, 'warmup_epochs', 5)))
        cosine_epochs = max(1, args.num_epochs - warmup_epochs)
        base_lrs = [group['lr'] for group in optimizer.param_groups]

        def make_lr_lambda(base_lr):
            min_ratio = min(1.0, args.min_lr / max(base_lr, 1e-12))

            def lr_lambda(epoch):
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return float(epoch + 1) / float(warmup_epochs)
                progress = min(1.0, float(epoch - warmup_epochs) / float(cosine_epochs))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1.0 - min_ratio) * cosine

            return lr_lambda

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            [make_lr_lambda(base_lr) for base_lr in base_lrs]
        )

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )


def apply_quality_preset(args):
    """应用质量评价通用训练预设。"""
    if getattr(args, 'quality_preset', 'none') != 'iqa_stable':
        return

    args.scheduler = 'warmup_cosine'
    args.text_lr_factor = 0.2
    args.image_lr_factor = 0.5
    args.loss_smooth_l1_weight = 0.55
    args.loss_pearson_weight = 0.20
    args.loss_rank_weight = 0.25
    args.rank_min_diff = 0.05
    args.rank_temperature = 0.5
    args.rank_loss_type = 'hard'
    args.train_aug_policy = 'legacy'
    args.use_ema = True
    args.ema_decay = 0.999
    args.use_balanced_sampling = True


def build_balanced_train_loader(dataloaders, args, base_logger):
    """按训练折MOS等宽分箱并使用逆频率采样，仅替换训练loader。"""
    if not getattr(args, 'use_balanced_sampling', False):
        log_and_print(base_logger, "质量分布均衡采样: disabled")
        return

    train_loader = dataloaders.get('train')
    if train_loader is None or not hasattr(train_loader, 'dataset'):
        log_and_print(base_logger, "质量分布均衡采样: 未找到训练集，跳过")
        return

    dataset = train_loader.dataset
    labels = np.asarray(dataset.label, dtype=np.float32)
    if labels.size == 0:
        log_and_print(base_logger, "质量分布均衡采样: 训练标签为空，跳过")
        return

    num_bins = max(2, int(getattr(args, 'num_quality_bins', 5)))
    label_min = float(labels.min())
    label_max = float(labels.max())
    if label_max - label_min < 1e-8:
        log_and_print(base_logger, "质量分布均衡采样: 标签范围过窄，跳过")
        return

    edges = np.linspace(label_min, label_max, num_bins + 1)
    bin_ids = np.digitize(labels, edges[1:-1], right=False)
    bin_counts = np.bincount(bin_ids, minlength=num_bins)
    sample_weights = np.asarray([1.0 / max(bin_counts[bin_id], 1) for bin_id in bin_ids], dtype=np.float64)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    dataloaders['train'] = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_loader.batch_size,
        sampler=sampler,
        shuffle=False,
        pin_memory=getattr(train_loader, 'pin_memory', True),
        drop_last=train_loader.drop_last,
        num_workers=train_loader.num_workers,
        persistent_workers=getattr(train_loader, 'persistent_workers', False),
        worker_init_fn=getattr(train_loader, 'worker_init_fn', None)
    )

    ranges = []
    for i in range(num_bins):
        ranges.append(f"[{edges[i]:.3f},{edges[i + 1]:.3f}]:{int(bin_counts[i])}")
    log_and_print(base_logger, f"质量分布均衡采样: enabled, bins={'; '.join(ranges)}")


QUALITY_DATASETS = ['AGIQA1K', 'AGIQA3K', 'AIGCIQA2023q', 'I2IQAq']


def get_quality_cv_dataloaders(args, dataset_name, fold):
    """加载质量评价任务的单折CV数据。"""
    if dataset_name == 'AGIQA1K':
        return get_AGIQA1K_CV_dataloaders(args, fold=fold)
    if dataset_name == 'AGIQA3K':
        return get_AGIQA3K_CV_dataloaders(args, fold=fold)
    if dataset_name == 'AIGCIQA2023q':
        args.cv_score_type = 'q'
        return get_AIGCIQA2023_CV_dataloaders(args, fold=fold, score_type='q')
    if dataset_name == 'I2IQAq':
        args.cv_score_type = 'q'
        return get_I2IQA_CV_dataloaders(args, fold=fold, score_type='q')
    raise ValueError(f"跨数据集实验仅支持质量任务: {QUALITY_DATASETS}, 当前: {dataset_name}")


def collect_full_quality_dataset(args, dataset_name, base_logger):
    """使用5个test fold合并去重，构建完整数据集字段。"""
    merged = {
        'images': [],
        'labels': [],
        'prompts': [],
        'image_names': []
    }
    seen = set()
    fold_counts = []

    for fold in range(1, 6):
        fold_loaders = get_quality_cv_dataloaders(args, dataset_name, fold)
        dataset = fold_loaders['val'].dataset
        added = 0
        for idx, image_name in enumerate(dataset.image_names):
            key = str(image_name)
            if key in seen:
                continue
            seen.add(key)
            merged['images'].append(dataset.image[idx])
            merged['labels'].append(float(dataset.label[idx]))
            merged['prompts'].append(dataset.text_prompt[idx])
            merged['image_names'].append(image_name)
            added += 1
        fold_counts.append(added)

    log_and_print(
        base_logger,
        f"跨库完整集 {dataset_name}: test folds去重后 n={len(merged['labels'])}, "
        f"fold新增={fold_counts}"
    )
    return merged


def build_cross_dataset_dataloaders(args, base_logger):
    """构建 train-on-source / test-on-target 的跨库dataloader。"""
    source = getattr(args, 'train_benchmark', '') or getattr(args, 'benchmark', '')
    target = getattr(args, 'test_benchmark', '')
    if not source or not target:
        raise ValueError("跨数据集实验需要同时指定 --train_benchmark 和 --test_benchmark")
    if source == target:
        raise ValueError("跨数据集实验要求 train_benchmark 与 test_benchmark 不同")
    if source not in QUALITY_DATASETS or target not in QUALITY_DATASETS:
        raise ValueError(f"跨数据集实验仅支持质量任务: {QUALITY_DATASETS}")

    log_and_print(base_logger, f"🔀 启用跨数据集实验: train={source}, test={target}")
    train_data = collect_full_quality_dataset(args, source, base_logger)
    test_data = collect_full_quality_dataset(args, target, base_logger)

    train_transforms, test_transforms = get_image_transforms(args)
    text_encoder_path = "clip_convnext" if getattr(args, 'text_encoder', '') == 'clip_convnext' else getattr(args, 'text_encoder_path', './deberta-v3-base')

    dataloaders = {
        'train': create_cv_dataloader(
            images=train_data['images'],
            labels=train_data['labels'],
            prompts=train_data['prompts'],
            transforms=train_transforms,
            text_encoder_path=text_encoder_path,
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=True,
            image_names=train_data['image_names']
        ),
        'val': create_cv_dataloader(
            images=test_data['images'],
            labels=test_data['labels'],
            prompts=test_data['prompts'],
            transforms=test_transforms,
            text_encoder_path=text_encoder_path,
            batch_size=args.test_batch_size,
            shuffle=False,
            drop_last=False,
            image_names=test_data['image_names']
        ),
        'dataset': f'{source}_to_{target}',
        'source_dataset': source,
        'target_dataset': target,
        'fold': 0
    }
    return dataloaders


def append_locked_csv(csv_path, header, row):
    """带文件锁追加CSV；已有文件缺字段时自动补表头。"""
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    with open(csv_path, 'a+', newline='') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames
            file_empty = existing_header is None
            existing_rows = list(reader) if not file_empty else []
            if file_empty:
                final_header = list(header)
            else:
                final_header = list(existing_header)
                for field in header:
                    if field not in final_header:
                        final_header.append(field)
            f.seek(0)
            f.truncate()
            writer = csv.DictWriter(f, fieldnames=final_header, extrasaction='ignore')
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow(existing_row)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

# 添加一个函数用于根据需要恢复原始分数范围
def restore_original_scores(scores, mapping_info):
    """
    将缩放后的分数恢复到原始范围，保持高精度
    
    参数:
        scores: 预测分数列表
        mapping_info: 从load_label返回的映射信息字典
    
    返回:
        恢复到原始范围的分数列表
    """
    original_min = mapping_info['original_min']
    original_max = mapping_info['original_max']
    target_min = mapping_info['target_min']
    target_max = mapping_info['target_max']
    
    # 进行逆向线性映射，使用高精度浮点数
    restored_scores = []
    for score in scores:
        # 确保分数在目标范围内
        clamped_score = max(float(target_min), min(float(target_max), float(score)))
        # 逆向映射: original = (normalized - target_min) / (target_max - target_min) * (original_max - original_min) + original_min
        original_score = (clamped_score - target_min) / (target_max - target_min) * (original_max - original_min) + original_min
        restored_scores.append(original_score)
    
    # 输出统计信息
    if len(restored_scores) > 0:
        min_score = min(restored_scores)
        max_score = max(restored_scores)
        mean_score = sum(restored_scores) / len(restored_scores)
        print(f"恢复后统计: 最小={min_score:.6f}, 最大={max_score:.6f}, 平均={mean_score:.6f}")
    
    return restored_scores


def get_label_stats(dataloaders, split):
    dataset = dataloaders[split].dataset
    labels = np.asarray(dataset.label, dtype=np.float32)
    if labels.size == 0:
        return {
            'mean': 0.0,
            'std': 1.0,
            'min': 0.0,
            'max': 0.0,
            'count': 0
        }
    return {
        'mean': float(labels.mean()),
        'std': float(labels.std(ddof=0)),
        'min': float(labels.min()),
        'max': float(labels.max()),
        'count': int(labels.size)
    }


def apply_train_zscore_label_norm(dataloaders, eps=1e-6):
    train_stats = get_label_stats(dataloaders, 'train')
    mean = train_stats['mean']
    std = max(train_stats['std'], eps)
    for split in ['train', 'val']:
        dataset = dataloaders[split].dataset
        dataset.label = [float((label - mean) / std) for label in dataset.label]
    return {
        'type': 'train_zscore',
        'mean': mean,
        'std': std
    }


def log_label_stats(base_logger, dataloaders, prefix):
    for split in ['train', 'val']:
        stats_info = get_label_stats(dataloaders, split)
        log_and_print(
            base_logger,
            f"{prefix}{split}标签统计: n={stats_info['count']}, "
            f"mean={stats_info['mean']:.6f}, std={stats_info['std']:.6f}, "
            f"range=[{stats_info['min']:.6f}, {stats_info['max']:.6f}]"
        )

if __name__ == '__main__':

    args = get_parser().parse_known_args()[0]

    # GPU环境变量已在文件开头设置，这里只需验证
    print(f"🔧 [确认] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # 验证GPU设置是否生效
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            all_gpus = result.stdout.strip().split('\n')
            print(f"🖥️  系统所有物理GPU: {len(all_gpus)} 个")
            for line in all_gpus:
                parts = line.split(', ')
                if len(parts) >= 2:
                    gpu_id, gpu_name = parts[0], parts[1]
                    print(f"    物理GPU {gpu_id}: {gpu_name}")
            
            # 检查选择的GPU是否存在
            gpu_exists = False
            for line in all_gpus:
                parts = line.split(', ')
                if len(parts) >= 2 and parts[0] == args.gpu:
                    gpu_exists = True
                    print(f"✅ 找到指定的物理GPU {args.gpu}: {parts[1]}")
                    break
            
            if not gpu_exists:
                print(f"❌ 错误: 指定的GPU {args.gpu} 不存在！")
                print(f"可用的GPU编号: {[line.split(', ')[0] for line in all_gpus if len(line.split(', ')) >= 2]}")
                exit(1)
                
        else:
            print("⚠️  无法获取GPU信息")
    except:
        print("⚠️  nvidia-smi命令不可用")

    # 现在设置CUDA设备，确保使用正确的GPU
    device = torch.device("cuda")  # 在设置环境变量之后定义device
    print(f"📱 PyTorch设备: {device} (对应物理GPU {args.gpu})")

    # 使用完整的确定性训练设置，统一使用run脚本的种子
    if args.seed is not None:
        setup_deterministic_training(args.seed)
    else:
        print("⚠️  警告: 未指定随机种子，使用默认种子224")
        setup_deterministic_training(224)

    # 添加GPU序列号到日志文件名避免冲突
    gpu_suffix = f"_gpu{args.gpu.replace(',', '-')}" if args.gpu else ""
    base_logger = get_logger(f'exp/FMAC{gpu_suffix}.log', args.log_info)

    # 添加随机种子输出
    log_and_print(base_logger, f"🌱 使用统一随机种子: {args.seed}")
    log_and_print(base_logger, f"🔒 确定性训练模式已启用，禁用CuDNN benchmark以确保可重复性")
    if getattr(args, 'cross_dataset', False):
        if not getattr(args, 'train_benchmark', '') or not getattr(args, 'test_benchmark', ''):
            raise ValueError("跨数据集实验需要指定 --train_benchmark 和 --test_benchmark")
        args.benchmark = args.train_benchmark
        args.use_cv = False
    apply_quality_preset(args)
    log_and_print(base_logger, f"质量任务预设: {args.quality_preset}")
    if args.quality_preset == 'iqa_stable':
        log_and_print(
            base_logger,
            "质量预设参数: "
            f"scheduler={args.scheduler}, warmup_epochs={args.warmup_epochs}, "
            f"text_lr_factor={args.text_lr_factor}, image_lr_factor={args.image_lr_factor}, "
            f"train_aug_policy={args.train_aug_policy}, use_ema={args.use_ema}, "
            f"ema_decay={args.ema_decay}, use_balanced_sampling={args.use_balanced_sampling}, "
            f"num_quality_bins={args.num_quality_bins}, "
            f"loss={args.loss_smooth_l1_weight}/{args.loss_pearson_weight}/{args.loss_rank_weight}, "
            f"rank_min_diff={args.rank_min_diff}, rank_temperature={args.rank_temperature}, "
            f"rank_loss_type={args.rank_loss_type}"
        )
    
    # 验证GPU使用情况并输出详细信息
    log_and_print(base_logger, f"🔧 已设置CUDA_VISIBLE_DEVICES={args.gpu}")
    
    # 检查可用的GPU数量
    gpu_count = torch.cuda.device_count()
    log_and_print(base_logger, f"📊 PyTorch可见GPU数量: {gpu_count}")
    
    # 验证CUDA设备可用性和基本信息
    if torch.cuda.is_available():
        log_and_print(base_logger, f"✅ CUDA可用状态: True")
        
        # 获取当前实际使用的GPU信息
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        log_and_print(base_logger, f"🖥️  当前使用GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        log_and_print(base_logger, f"🎯 物理GPU编号: {args.gpu}")
        log_and_print(base_logger, f"📱 PyTorch设备编号: cuda:{current_device}")
        
        # 验证是否真的在使用指定的GPU（通过创建一个小tensor测试）
        test_tensor = torch.tensor([1.0], device=f'cuda:{current_device}')
        log_and_print(base_logger, f"✅ GPU {args.gpu} 测试成功，张量创建在: {test_tensor.device}")
        
    else:
        log_and_print(base_logger, "⚠️  警告: CUDA不可用，训练将失败！")

    # 打印模型信息
    log_and_print(base_logger, f"Using backbone: {args.backbone}")
    log_and_print(base_logger, f"Text encoder: {args.text_encoder}")
    log_and_print(base_logger, f"Text LR factor: {args.text_lr_factor}, Image LR factor: {args.image_lr_factor}")
    # log_and_print(base_logger, f"Contrastive weight: {args.contrastive_weight}")
    log_and_print(base_logger, f"学习率调度器: {args.scheduler}")
    
    # 打印频域特征相关配置
    log_and_print(base_logger, f"使用频域特征: {args.use_frequency_features}")
    if args.use_frequency_features:
        log_and_print(base_logger, f"频域特征维度: {args.freq_feature_dim}")
        log_and_print(base_logger, "使用增强版FFT频域特征提取器")
    
    # 设置文本编码器路径
    text_encoder_path = args.text_encoder_path
    if args.text_encoder == 'deberta':
        if 'deberta' not in text_encoder_path.lower():
            if os.path.exists('./deberta-v3-base'):
                text_encoder_path = './deberta-v3-base'
            elif os.path.exists('./deberta-base'):
                text_encoder_path = './deberta-base'
            else:
                log_and_print(base_logger, "警告: 未找到DeBERTa模型路径，将使用指定的路径")
    elif args.text_encoder == 'roberta':
        if 'roberta' not in text_encoder_path.lower():
            if os.path.exists('./roberta-base'):
                text_encoder_path = './roberta-base'
            else:
                log_and_print(base_logger, "警告: 未找到RoBERTa模型路径，将使用指定的路径")
    elif args.text_encoder == 'clip_convnext':
        # CLIPConvNeXt文本编码器不需要本地路径，使用特殊标识符
        text_encoder_path = "clip_convnext"
        log_and_print(base_logger, "使用CLIP-ConvNeXt文本编码器")
    
    log_and_print(base_logger, f"使用文本编码器路径: {text_encoder_path}")

    normalize_to_five_flag = args.normalize_scores_to_five
    
    # 初始化backbone和编码器 - 只保留性能最好的两个backbone
    if args.backbone == 'swin':
        SwinExtractor = get_swin_extractor()
        backbone = SwinExtractor().to(device)
        encoder = Encoder(backbone, model_path=text_encoder_path, 
                          freq_feature_dim=args.freq_feature_dim, 
                          use_frequency_features=args.use_frequency_features,
                          use_enhanced_fusion=args.use_enhanced_fusion,
                          fusion_type=args.fusion_type,
                          num_heads=args.num_heads,
                          dataset_name=args.benchmark,
                          use_freq_enhanced_fusion=args.use_freq_enhanced_fusion,
                          freq_enhancement_weight=args.freq_enhancement_weight,
                          contrastive_temperature=args.contrastive_temperature).to(device)
        regressor = MLP(
            1024,
            model_path=text_encoder_path,
            dataset_name=args.benchmark,
            freq_feature_dim=args.freq_feature_dim,
            normalize_to_five=normalize_to_five_flag,
            txt_dim=encoder.txt_dim
        ).to(device)

    elif args.backbone == 'clipconvnext':
        if not CLIPCONVNEXT_AVAILABLE:
            raise ValueError("CLIP-ConvNeXt编码器不可用，请安装open_clip_torch: pip install open_clip_torch>=2.20.0")
        
        # 使用CLIP-ConvNeXt编码器
        backbone = get_clip_convnext_encoder(
            model_name='convnext_large_d_320',
            pretrained='laion2b_s29b_b131k_ft',
            output_dim=None,  # 使用默认维度（动态检测）
            freeze_encoder=False,  # 可训练模式
            use_enhanced=False,  # 使用标准版本
            use_trunk_features=False  # 使用CLIP投影语义向量，保持图文特征处于同一空间
        ).to(device)
        
        log_and_print(base_logger, f"初始化CLIP-ConvNeXt编码器: convnext_large_d_320")
        log_and_print(base_logger, f"特征维度: {backbone.get_feature_dim()}")
        
        encoder = Encoder(backbone, model_path=text_encoder_path,
                          freq_feature_dim=args.freq_feature_dim,
                          use_frequency_features=args.use_frequency_features,
                          use_enhanced_fusion=args.use_enhanced_fusion,
                          fusion_type=args.fusion_type,
                          num_heads=args.num_heads,
                          dataset_name=args.benchmark,
                          use_freq_enhanced_fusion=args.use_freq_enhanced_fusion,
                          freq_enhancement_weight=args.freq_enhancement_weight,
                          contrastive_temperature=args.contrastive_temperature,
                          ablation_mode=args.ablation_mode,
                          disable_freq_clip_fusion=args.disable_freq_clip_fusion,
                          disable_mfb_fusion=args.disable_mfb_fusion).to(device)
        regressor = MLP(
            backbone.get_feature_dim(),
            model_path=text_encoder_path,
            dataset_name=args.benchmark,
            freq_feature_dim=args.freq_feature_dim,
            normalize_to_five=normalize_to_five_flag,
            txt_dim=encoder.txt_dim,
            ablation_mode=args.ablation_mode
        ).to(device)
    else:
        # 默认使用Swin Transformer
        log_and_print(base_logger, f"未识别的backbone '{args.backbone}'，使用默认的Swin Transformer")
        SwinExtractor = get_swin_extractor()
        backbone = SwinExtractor().to(device)
        encoder = Encoder(backbone, model_path=text_encoder_path,
                          freq_feature_dim=args.freq_feature_dim,
                          use_frequency_features=args.use_frequency_features,
                          use_enhanced_fusion=args.use_enhanced_fusion,
                          fusion_type=args.fusion_type,
                          num_heads=args.num_heads,
                          dataset_name=args.benchmark,
                          use_freq_enhanced_fusion=args.use_freq_enhanced_fusion,
                          freq_enhancement_weight=args.freq_enhancement_weight,
                          contrastive_temperature=args.contrastive_temperature,
                          ablation_mode=args.ablation_mode,
                          disable_freq_clip_fusion=args.disable_freq_clip_fusion,
                          disable_mfb_fusion=args.disable_mfb_fusion).to(device)
        regressor = MLP(
            1024,
            model_path=text_encoder_path,
            dataset_name=args.benchmark,
            freq_feature_dim=args.freq_feature_dim,
            normalize_to_five=normalize_to_five_flag,
            txt_dim=encoder.txt_dim
        ).to(device)

    # 使用DataParallel进行多GPU训练 - 仅在指定多个GPU时启用
    gpu_list = args.gpu.split(',')
    use_multi_gpu = len(gpu_list) > 1 and torch.cuda.device_count() > 1
    
    if use_multi_gpu:
        log_and_print(base_logger, f"使用DataParallel进行多GPU训练，GPU数量: {len(gpu_list)}")
        encoder = nn.DataParallel(encoder)
        regressor = nn.DataParallel(regressor)
        # 调整学习率以适应多GPU训练
        original_lr = args.lr
        args.lr = args.lr * len(gpu_list)
        log_and_print(base_logger, f"调整学习率: {original_lr} -> {args.lr} (x{len(gpu_list)})")
    else:
        log_and_print(base_logger, "使用单GPU训练")

    is_cross_dataset = getattr(args, 'cross_dataset', False)
    is_official_split = args.benchmark == 'AIGCIQA20K'
    cross_source = getattr(args, 'train_benchmark', '') if is_cross_dataset else ''
    cross_target = getattr(args, 'test_benchmark', '') if is_cross_dataset else ''
    is_ablation_run = (
        str(getattr(args, 'run_id', '')).startswith('ablation_')
        or str(getattr(args, 'log_info', '')).startswith('ablation_')
    )

    # 数据加载器选择逻辑
    if is_cross_dataset:
        dataloaders = build_cross_dataset_dataloaders(args, base_logger)
        log_and_print(base_logger, "✅ 跨数据集数据加载器创建完成")
    elif is_official_split:
        log_and_print(base_logger, "🔄 启用AIGCIQA20K官方train/val/test划分")
        dataloaders = get_AIGCIQA20K_dataloaders(args)
        log_and_print(base_logger, "✅ AIGCIQA20K官方划分数据加载器创建完成")
    elif args.use_cv:
        # 使用交叉验证数据加载器
        log_and_print(base_logger, f"🔄 启用交叉验证模式 - 第 {args.fold} 折")
        
        log_and_print(base_logger, f"📊 数据集: {args.benchmark}")
        if args.benchmark in ['AIGCIQA2023q', 'AIGCIQA2023a', 'AIGCIQA2023c']:
            if args.benchmark == 'AIGCIQA2023q':
                score_type = 'q'
            elif args.benchmark == 'AIGCIQA2023a':
                score_type = 'a'
            elif args.benchmark == 'AIGCIQA2023c':
                score_type = 'c'
            else:
                score_type = args.cv_score_type
            args.cv_score_type = score_type
            log_and_print(base_logger, f"使用评分类型: {score_type}")
            dataloaders = get_AIGCIQA2023_CV_dataloaders(args, fold=args.fold, score_type=score_type)
        elif args.benchmark in ['I2IQAq', 'I2IQAa', 'I2IQAc']:
            if args.benchmark == 'I2IQAq':
                score_type = 'q'
            elif args.benchmark == 'I2IQAa':
                score_type = 'a'
            else:
                score_type = 'c'
            args.cv_score_type = score_type
            log_and_print(base_logger, f"使用I2IQA评分类型: {score_type}")
            dataloaders = get_I2IQA_CV_dataloaders(args, fold=args.fold, score_type=score_type)
        elif args.benchmark == 'AGIQA3K':
            dataloaders = get_AGIQA3K_CV_dataloaders(args, fold=args.fold)
        elif args.benchmark == 'AGIQA3Kc':
            dataloaders = get_AGIQA3Kc_CV_dataloaders(args, fold=args.fold)
        elif args.benchmark == 'AGIQA1K':
            dataloaders = get_AGIQA1K_CV_dataloaders(args, fold=args.fold)
        else:
            log_and_print(base_logger, f"⚠️  未知的数据集: {args.benchmark}，使用AGIQA-1K交叉验证数据加载器")
            dataloaders = get_AGIQA1K_CV_dataloaders(args, fold=args.fold)
        
        log_and_print(base_logger, f"✅ 交叉验证数据加载器创建完成")
    else:
        # 非交叉验证模式仅用于带官方划分的数据集
        raise ValueError("当前版本只支持交叉验证模式，或AIGCIQA20K官方划分。请设置 --use_cv 参数")

    log_label_stats(base_logger, dataloaders, "原始")
    label_norm_info = {'type': 'none', 'mean': 0.0, 'std': 1.0}
    if args.label_norm == 'train_zscore':
        label_norm_info = apply_train_zscore_label_norm(dataloaders)
        log_and_print(
            base_logger,
            f"启用训练折标签z-score: mean={label_norm_info['mean']:.6f}, std={label_norm_info['std']:.6f}"
        )
        log_label_stats(base_logger, dataloaders, "归一化后")
    else:
        log_and_print(base_logger, "标签归一化: none")
    build_balanced_train_loader(dataloaders, args, base_logger)

    if args.freeze_text_encoder:
        text_module = encoder.module.text_encoder if hasattr(encoder, 'module') else encoder.text_encoder
        for param in text_module.parameters():
            param.requires_grad = False
        log_and_print(base_logger, "已冻结文本编码器参数")

    # 一致性任务标记（仅这些任务启用对比损失）
    is_correspondence_task = args.benchmark in ['AIGCIQA2023c', 'AGIQA3Kc', 'I2IQAc']
    if is_correspondence_task:
        if args.use_global_contrastive:
            log_and_print(base_logger, f"一致性任务启用全局对比损失: weight={args.global_contrastive_weight}, temp={args.contrastive_temperature}")
        else:
            log_and_print(base_logger, "一致性任务未启用全局对比损失")
    else:
        if args.use_global_contrastive:
            log_and_print(base_logger, "非一致性任务忽略全局对比损失设置")

    # 主回归损失函数：SRCC优先的SmoothL1 + Pearson + PairwiseRank
    log_and_print(
        base_logger,
        "使用SRCC优先组合损失: "
        f"{args.loss_smooth_l1_weight:.2f}*SmoothL1 + "
        f"{args.loss_pearson_weight:.2f}*PearsonLoss + "
        f"{args.loss_rank_weight:.2f}*PairwiseRankLoss "
        f"(rank_type={args.rank_loss_type}, rank_min_diff={args.rank_min_diff}, "
        f"rank_temperature={args.rank_temperature})"
    )

    # 忽略旧创新损失相关开关，避免与当前稳定组合损失重复叠加
    if getattr(args, 'use_innovative_loss', False):
        log_and_print(base_logger, "提示: use_innovative_loss 已被忽略（当前版本使用稳定组合质量损失）")
        args.use_innovative_loss = False
    
    # 使用优化后的优化器和调度器
    optimizer = get_optimizer_and_scheduler(encoder, regressor, args, use_multi_gpu)
    
    # 记录优化器和梯度累积配置
    log_and_print(base_logger, f"使用AdamW优化器，梯度累积步数: {args.gradient_accumulation_steps}")
    for group in optimizer.param_groups:
        log_and_print(
            base_logger,
            f"参数组 {group.get('name', 'group')}: lr={group['lr']:.2e}, "
            f"weight_decay={group['weight_decay']}, params={group.get('num_params', 0)}"
        )
    log_and_print(base_logger, f"梯度裁剪阈值: 1.0 (更保守的设置)")

    # 初始化高级训练策略组件
    if args.use_advanced_training and ADVANCED_TRAINING_AVAILABLE:
        log_and_print(base_logger, "启用第四阶段高级训练策略")
        
        # Mixup数据增强
        if args.use_mixup:
            mixup = MixupAugmentation(alpha=args.mixup_alpha)
            log_and_print(base_logger, f"启用Mixup数据增强，alpha: {args.mixup_alpha}")
        else:
            mixup = None
        
        # 梯度累积
        if args.gradient_accumulation_steps > 1:
            grad_accumulator = GradientAccumulation(args.gradient_accumulation_steps)
            log_and_print(base_logger, f"启用梯度累积，累积步数: {args.gradient_accumulation_steps}")
        else:
            grad_accumulator = None
        
        # 渐进式学习
        if args.use_progressive_learning:
            progressive_learning = ProgressiveLearning(warmup_epochs=args.progressive_warmup_epochs)
            log_and_print(base_logger, f"启用渐进式学习，预热epochs: {args.progressive_warmup_epochs}")
        else:
            progressive_learning = None
        
        # 自适应损失权重
        if args.use_adaptive_loss_weighting and args.use_innovative_loss:
            adaptive_loss_weighting = AdaptiveLossWeighting(num_losses=5)  # 5种损失类型
            log_and_print(base_logger, "启用自适应损失权重调整")
        else:
            adaptive_loss_weighting = None
        
    else:
        mixup = None
        grad_accumulator = None
        progressive_learning = None
        adaptive_loss_weighting = None
        log_and_print(base_logger, "使用标准训练策略")

    if args.use_ema:
        ema_encoder = ModelEMA(encoder, decay=args.ema_decay)
        ema_regressor = ModelEMA(regressor, decay=args.ema_decay)
        log_and_print(base_logger, f"启用内置EMA，衰减率: {args.ema_decay}")
    else:
        ema_encoder = None
        ema_regressor = None

    # 训练参数
    scheduler = build_scheduler(optimizer, args)
    t_max = args.t_max if args.t_max is not None else args.num_epochs
    if args.scheduler == 'cosine':
        log_and_print(base_logger, f"使用余弦退火学习率调度器, T_max={t_max}, min_lr={args.min_lr}")
    elif args.scheduler == 'warmup_cosine':
        log_and_print(base_logger, f"使用Warmup+Cosine学习率调度器, warmup_epochs={args.warmup_epochs}, min_lr={args.min_lr}")
    else:
        log_and_print(base_logger, "使用ReduceLROnPlateau学习率调度器")

    # 初始化混合精度训练
    scaler = torch.cuda.amp.GradScaler()  # 用于混合精度训练
    log_and_print(base_logger, "启用混合精度训练")

    # 训练指标记录
    epoch_srcc_best = 0  # SRCC最佳时的epoch
    epoch_plcc_best = 0
    epoch_krcc_best = 0
    rho_s_best = 0.0
    rho_p_best = 0.0
    rho_k_best = 0.0  # KRCC最佳值
    current_train_metrics = {'srcc': 0.0, 'plcc': 0.0, 'krcc': 0.0}
    best_train_metrics = {'srcc': 0.0, 'plcc': 0.0, 'krcc': 0.0}

    # 创建保存目录
    save_dir = os.path.join('ckpts')
    os.makedirs(save_dir, exist_ok=True)

    # 最优模型保存起始epoch控制（人类计数，从1开始；内部转为0基索引）
    save_best_from_epoch = getattr(args, 'save_best_start_epoch', 1)
    save_best_from_epoch_idx = max(0, int(save_best_from_epoch) - 1)
    log_and_print(base_logger, f"从第 {save_best_from_epoch} 个epoch开始保存最佳模型")

    for epoch in range(args.num_epochs):
        log_and_print(base_logger, f'Epoch: {epoch}')

        # GPU性能监控 - 只在第一个epoch开始时检查一次
        if epoch == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()  # 确保所有GPU操作完成
            start_gpu_mem = torch.cuda.memory_allocated()
            log_and_print(base_logger, f"GPU内存使用: {start_gpu_mem / 1024**2:.2f} MB")
            
            # 清理未使用的缓存以释放内存
            torch.cuda.empty_cache()
            log_and_print(base_logger, f"清理缓存后GPU内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # 只进行训练和验证，不进行测试
        for split in ['train', 'val']:
            true_scores = []
            pred_scores = []
            epoch_loss = 0.0  # 记录整个epoch的总损失
            epoch_smooth_l1 = 0.0
            epoch_pearson = 0.0
            epoch_rank = 0.0
            batch_count = 0  # 批次计数器

            if split == 'train':
                encoder.train()
                regressor.train()
                torch.set_grad_enabled(True)
                
                # 找到初始GPU内存监控部分，将其替换为更安全的版本
                if epoch == 0:
                    if torch.cuda.is_available():
                        log_and_print(base_logger, f"初始GPU内存分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                        log_and_print(base_logger, f"初始GPU内存缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                        
                        # 移除可能导致维度不匹配的dummy测试
                        log_and_print(base_logger, "跳过dummy测试以避免维度不匹配")
                        log_and_print(base_logger, f"模型参数位置: {next(encoder.parameters()).device} (物理GPU {args.gpu})")
            else:
                encoder.eval()
                regressor.eval()
                torch.set_grad_enabled(False)
                
                # 在验证时使用EMA模型（如果启用）
                if ema_encoder is not None and ema_regressor is not None:
                    ema_encoder.apply_shadow()
                    ema_regressor.apply_shadow()

            for data in tqdm(dataloaders[split]):
                try:
                    # 确保数据不为空
                    if len(data['MOS_score']) == 0:
                        continue
                        
                    # 检查批次大小，避免BatchNorm问题
                    if len(data['MOS_score']) == 1:
                        if DEBUG_MODE and epoch == 0 and batch_count == 0:
                            print("跳过批次大小为1的样本，避免BatchNorm错误")
                        continue
                        
                    true_scores.extend(data['MOS_score'].numpy())

                    image = data['image'].to(device)  # B, C, H, W
                    text_prompt = data['prompt'].to(device)
                    mask = data['attention_mask'].to(device)
                    mos_score = data['MOS_score'].float().to(device)
                    
                    # 应用Mixup数据增强（如果启用）
                    if split == 'train' and mixup is not None:
                        # 首先获取文本特征用于mixup
                        with torch.no_grad():
                            temp_feature, temp_text_features, temp_image_features = encoder(image, text_prompt, mask, batch_count=batch_count, epoch=epoch)
                        
                        mixed_images, mixed_labels, mixed_text_features, lam, index = mixup(
                            image, mos_score, temp_text_features
                        )
                        
                        # 使用混合后的数据
                        image = mixed_images
                        mos_score = mixed_labels
                        # 注意：文本特征已经在前向传播中混合了
                    
                    # 仅在DEBUG_MODE时打印批次信息，且只在第一个epoch的第一个批次
                    if DEBUG_MODE and epoch == 0 and batch_count == 0:
                        print(f"批次形状: image={image.shape}, text={text_prompt.shape}, mask={mask.shape}")
                        print(f"批次标签: {mos_score}")
                    
                    # 尝试执行前向传播，传递batch_count和epoch参数
                    # [AMP修复] 将编码器/回归器前向放入autocast，使混合精度真正作用于主要计算
                    with torch.cuda.amp.autocast(enabled=(split == 'train')):
                        feature, text_features, image_features = encoder(image, text_prompt, mask, batch_count=batch_count, epoch=epoch)
                        preds = regressor(feature).view(-1)

                    # 在添加到pred_scores前确保preds不为空
                    if len(preds) > 0:
                        # 直接使用原始预测值，不进行任何缩放
                        pred_scores.extend([i.item() for i in preds])

                        # 只在第0个epoch的第一个批次输出详细信息
                        if epoch == 0 and batch_count == 0:
                            print(f"\n{split}集(Epoch {epoch}):")
                            print(f"预测值范围: {preds.min().item():.4f}-{preds.max().item():.4f}")
                            print(f"真实值范围: {mos_score.min().item():.4f}-{mos_score.max().item():.4f}")

                            # 打印前5个样本的对比
                            for i in range(min(5, len(preds))):
                                print(f"样本{i+1}: 预测={preds[i].item():.2f}, 真实={mos_score[i].item():.2f}")
                    else:
                        if DEBUG_MODE and epoch == 0 and batch_count == 0:
                            print("警告: 预测结果为空")
                        continue

                    if split == 'train':
                        with torch.cuda.amp.autocast():
                            loss, smooth_l1_loss, pearson_loss, rank_loss = compute_quality_loss(
                                preds,
                                mos_score,
                                smooth_l1_weight=args.loss_smooth_l1_weight,
                                pearson_weight=args.loss_pearson_weight,
                                rank_weight=args.loss_rank_weight,
                                rank_min_diff=args.rank_min_diff,
                                rank_temperature=args.rank_temperature,
                                rank_loss_type=args.rank_loss_type
                            )
                            if (DEBUG_MODE or args.log_training_diagnostics) and epoch == 0 and batch_count < 1:
                                batch_label_std = mos_score.detach().float().std(unbiased=False).item()
                                hard_pair_ratio = compute_rank_pair_ratio(mos_score.detach(), args.rank_min_diff)
                                log_and_print(
                                    base_logger,
                                    f"训练诊断: batch_label_std={batch_label_std:.6f}, "
                                    f"hard_rank_pair_ratio={hard_pair_ratio:.4f}, "
                                    f"total={loss.item():.6f}, smooth_l1={smooth_l1_loss.item():.6f}, "
                                    f"pearson={pearson_loss.item():.6f}, rank={rank_loss.item():.6f}"
                                )
                            if is_correspondence_task and args.use_global_contrastive:
                                contrastive_loss = compute_global_contrastive_loss(
                                    image_features, text_features, args.contrastive_temperature
                                )
                                loss = loss + args.global_contrastive_weight * contrastive_loss
                                if DEBUG_MODE and epoch == 0 and batch_count < 1:
                                    print(f"对比损失: {contrastive_loss.item():.6f} (权重={args.global_contrastive_weight})")

                        epoch_loss += loss.item()
                        epoch_smooth_l1 += smooth_l1_loss.item()
                        epoch_pearson += pearson_loss.item()
                        epoch_rank += rank_loss.item()
                        batch_count += 1
                        
                        if grad_accumulator is not None:
                            loss = grad_accumulator.scale_loss(loss)
                        
                        if epoch == 0 and batch_count == 1 and torch.cuda.is_available():
                            log_and_print(base_logger, f"批次1后GPU内存分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                            log_and_print(base_logger, f"批次1后GPU内存缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                            log_and_print(base_logger, f"图像数据在GPU上: {image.device.type == 'cuda'}")
                            log_and_print(base_logger, f"文本数据在GPU上: {text_prompt.device.type == 'cuda'}")
                            log_and_print(base_logger, f"模型输出在GPU上: {preds.device.type == 'cuda'}")
                        
                        # 优化的梯度处理逻辑 - 适合小数据集的标准训练
                        if args.gradient_accumulation_steps == 1:
                            # 标准训练：每个批次都清零梯度和更新参数（推荐用于小数据集）
                            optimizer.zero_grad(set_to_none=True)
                            
                            # 使用scaler进行反向传播
                            scaler.scale(loss).backward()
                            
                            # 梯度裁剪
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_(regressor.parameters(), max_norm=1.0)
                            
                            # 参数更新
                            scaler.step(optimizer)
                            scaler.update()
                            
                            # 更新EMA参数（如果启用）
                            if ema_encoder is not None:
                                ema_encoder.update()
                            if ema_regressor is not None:
                                ema_regressor.update()
                        else:
                            # 梯度累积训练：用于大batch_size模拟（不推荐用于小数据集）
                            if (batch_count - 1) % args.gradient_accumulation_steps == 0:
                                optimizer.zero_grad(set_to_none=True)
                            
                            # 根据梯度累积步数缩放损失
                            scaled_loss = loss / args.gradient_accumulation_steps
                            
                            # 使用scaler进行反向传播
                            scaler.scale(scaled_loss).backward()
                            
                            # 参数更新（每accumulation_steps个批次更新一次）
                            if batch_count % args.gradient_accumulation_steps == 0:
                                # 只在需要更新参数时才调用unscale_和梯度裁剪
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                                torch.nn.utils.clip_grad_norm_(regressor.parameters(), max_norm=1.0)
                                
                                scaler.step(optimizer)
                                scaler.update()
                                
                                # 更新EMA参数（如果启用）
                                if ema_encoder is not None:
                                    ema_encoder.update()
                                if ema_regressor is not None:
                                    ema_regressor.update()
                except Exception as e:
                    # 打印详细的错误信息
                    print(f"处理批次时发生错误: {e}")
                    try:
                        print(f"数据形状: image={image.shape}, text_prompt={text_prompt.shape}, mask={mask.shape}")
                    except:
                        print("无法打印数据形状")
                    
                    # 如果是CUDA OOM错误，清理缓存并跳过当前批次
                    if "CUDA out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print("清理GPU缓存后继续训练")
                    
                    continue

            # 计算并显示平均损失
            if (
                split == 'train'
                and args.gradient_accumulation_steps > 1
                and batch_count > 0
                and batch_count % args.gradient_accumulation_steps != 0
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(regressor.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if ema_encoder is not None:
                    ema_encoder.update()
                if ema_regressor is not None:
                    ema_regressor.update()

            if split == 'train' and batch_count > 0:
                avg_loss = epoch_loss / batch_count
                avg_smooth_l1 = epoch_smooth_l1 / batch_count
                avg_pearson = epoch_pearson / batch_count
                avg_rank = epoch_rank / batch_count
                log_and_print(
                    base_logger,
                    f'📈 训练损失: total={avg_loss:.4f}, smooth_l1={avg_smooth_l1:.4f}, pearson={avg_pearson:.4f}, rank={avg_rank:.4f}'
                )

            # 在评估指标计算前修改预测分数，如果需要恢复原始范围
            # 确保true_scores和pred_scores不为空且长度相同
            if len(true_scores) == 0 or len(pred_scores) == 0:
                print(f"警告: {split}集true_scores或pred_scores为空")
                if split == 'train':
                    rho_s = 0.0
                    rho_p = 0.0
                    rho_k = 0.0  # KRCC初始化为0
                else:
                    # 验证集上如果没有预测，需要特殊处理
                    print(f"错误: 验证集上没有有效预测")
                    rho_s = -1.0
                    rho_p = -1.0
                    rho_k = -1.0  # KRCC初始化为-1
            elif len(true_scores) != len(pred_scores):
                print(f"警告: {split}集样本数量不匹配，截断至相同长度")
                min_len = min(len(true_scores), len(pred_scores))
                true_scores = true_scores[:min_len]
                pred_scores = pred_scores[:min_len]
                rho_s = stats.spearmanr(true_scores, pred_scores)[0]
                rho_p = stats.pearsonr(true_scores, pred_scores)[0]
                rho_k = kendalltau(true_scores, pred_scores)[0]  # 计算KRCC
            else:
                # 正常情况 - 计算评价指标
                rho_s = stats.spearmanr(true_scores, pred_scores)[0]
                rho_p = stats.pearsonr(true_scores, pred_scores)[0]
                rho_k = kendalltau(true_scores, pred_scores)[0]  # 计算KRCC
                
                # 打印验证集的统计信息
                if split == 'val':
                    # 计算分数统计
                    min_true = min(true_scores)
                    max_true = max(true_scores)
                    mean_true = sum(true_scores) / len(true_scores)
                    min_pred = min(pred_scores)
                    max_pred = max(pred_scores)
                    mean_pred = sum(pred_scores) / len(pred_scores)
                    
                    print("\n验证集分数统计:")
                    print(f"真实值范围: {min_true:.6f} - {max_true:.6f}, 平均值: {mean_true:.6f}")
                    print(f"预测值范围: {min_pred:.6f} - {max_pred:.6f}, 平均值: {mean_pred:.6f}")
                    
                    print("\n验证集前5个样本:")
                    for i in range(min(5, len(true_scores))):
                        print(f"样本 {i+1}: 预测值={pred_scores[i]:.6f}, 真实值={true_scores[i]:.6f}, 差值={pred_scores[i]-true_scores[i]:.6f}")
                    
                    # 打印相关系数
                    print(f"\n验证集相关系数:")
                    print(f"SRCC: {rho_s:.6f}, PLCC: {rho_p:.6f}, KRCC: {rho_k:.6f}")
                    print("="*50)

            log_and_print(base_logger, f'{split} spearmanr_correlation: {rho_s:.4f}, pearsonr_correlation: {rho_p:.4f}, kendallr_correlation: {rho_k:.4f}')
            if split == 'train':
                current_train_metrics = {
                    'srcc': float(rho_s),
                    'plcc': float(rho_p),
                    'krcc': float(rho_k)
                }

            # 在验证集上更新最佳模型
            if split == 'val':
                improved = rho_s > rho_s_best
                if improved:
                    # 始终更新最佳指标记录（无论是否保存权重）
                    rho_s_best = rho_s
                    rho_p_best = rho_p
                    rho_k_best = rho_k  # 更新KRCC最佳值
                    epoch_srcc_best = epoch
                    epoch_plcc_best = epoch
                    epoch_krcc_best = epoch
                    best_train_metrics = current_train_metrics.copy()

                    # 输出最佳指标日志，包含fold信息
                    if is_cross_dataset:
                        fold_info = f" ({cross_source} -> {cross_target})"
                    else:
                        fold_info = f" (Fold {args.fold})" if hasattr(args, 'use_cv') and args.use_cv and hasattr(args, 'fold') else ""
                    log_and_print(base_logger, f' EPOCH_best: {epoch}, SRCC_best: {rho_s_best:.6f}, PLCC_best: {rho_p_best:.6f}, KRCC_best: {rho_k_best:.6f}{fold_info}')

                    # 仅在达到起始epoch后才执行保存（且未禁用保存）
                    if epoch >= save_best_from_epoch_idx and not getattr(args, 'no_save_model', False):
                        # 保存最佳模型 - 支持交叉验证fold编号
                        if is_cross_dataset:
                            save_path = os.path.join(save_dir, f'best_model_cross_{cross_source}_to_{cross_target}.pth')
                        elif hasattr(args, 'use_cv') and args.use_cv and hasattr(args, 'fold'):
                            # 交叉验证模式：消融实验额外包含ablation_mode，避免并行任务覆盖checkpoint
                            if is_ablation_run:
                                save_path = os.path.join(
                                    save_dir,
                                    f'best_model_{args.benchmark}_{args.ablation_mode}_fold{args.fold}.pth'
                                )
                            else:
                                save_path = os.path.join(save_dir, f'best_model_{args.benchmark}_fold{args.fold}.pth')
                        else:
                            # 标准模式：不包含fold编号
                            save_path = os.path.join(save_dir, f'best_model_{args.benchmark}.pth')

                        # 仅保存模型权重和指标，避免AdamW优化器状态让checkpoint膨胀到数GB
                        checkpoint_meta = {
                            'epoch': epoch,
                            'rho_s': rho_s,
                            'rho_p': rho_p,
                            'rho_k': rho_k,
                            'fold': 0 if (is_cross_dataset or is_official_split) else getattr(args, 'fold', None),
                            'score_type': getattr(args, 'cv_score_type', None),
                            'run_id': getattr(args, 'run_id', ''),
                            'ablation_mode': getattr(args, 'ablation_mode', 'full'),
                            'benchmark': getattr(args, 'benchmark', ''),
                            'source_dataset': cross_source,
                            'target_dataset': cross_target,
                            'cross_dataset': is_cross_dataset,
                            'quality_preset': getattr(args, 'quality_preset', 'none'),
                            'scheduler': getattr(args, 'scheduler', ''),
                            'warmup_epochs': getattr(args, 'warmup_epochs', 0),
                            'train_aug_policy': getattr(args, 'train_aug_policy', 'legacy'),
                            'use_ema': getattr(args, 'use_ema', False),
                            'ema_decay': getattr(args, 'ema_decay', None),
                            'loss_smooth_l1_weight': getattr(args, 'loss_smooth_l1_weight', None),
                            'loss_pearson_weight': getattr(args, 'loss_pearson_weight', None),
                            'loss_rank_weight': getattr(args, 'loss_rank_weight', None),
                            'rank_min_diff': getattr(args, 'rank_min_diff', None),
                            'rank_temperature': getattr(args, 'rank_temperature', None),
                            'rank_loss_type': getattr(args, 'rank_loss_type', None),
                            'label_norm': label_norm_info['type'],
                            'label_norm_mean': label_norm_info['mean'],
                            'label_norm_std': label_norm_info['std'],
                            'freeze_text_encoder': getattr(args, 'freeze_text_encoder', False),
                            'train_rho_s': best_train_metrics['srcc'],
                            'train_rho_p': best_train_metrics['plcc'],
                            'train_rho_k': best_train_metrics['krcc'],
                            'generalization_gap_srcc': best_train_metrics['srcc'] - rho_s,
                            'use_balanced_sampling': getattr(args, 'use_balanced_sampling', False),
                            'num_quality_bins': getattr(args, 'num_quality_bins', 5),
                            'cv_protocol': 'cross-dataset-full-test-folds' if is_cross_dataset else (
                                'official-train-val-test' if is_official_split else '5-fold-cv_folds'
                            )
                        }
                        if use_multi_gpu:
                            checkpoint = {
                                'encoder_state_dict': encoder.module.state_dict(),
                                'regressor_state_dict': regressor.module.state_dict(),
                                **checkpoint_meta
                            }
                        else:
                            checkpoint = {
                                'encoder_state_dict': encoder.state_dict(),
                                'regressor_state_dict': regressor.state_dict(),
                                **checkpoint_meta
                            }

                        torch.save(checkpoint, save_path)
                        log_and_print(base_logger, f'✅ 模型已保存: {save_path}')

            # 如果使用了EMA，在验证后恢复原始参数
            if split != 'train' and ema_encoder is not None and ema_regressor is not None:
                ema_encoder.restore()
                ema_regressor.restore()

        # 更新学习率调度器
        if args.scheduler in ['cosine', 'warmup_cosine']:
            scheduler.step()
            lr_parts = []
            for pg in optimizer.param_groups:
                lr_parts.append(f"{pg.get('name', 'group')}={pg['lr']:.2e}")
            log_and_print(base_logger, "当前学习率: " + ", ".join(lr_parts))
        else:
            scheduler.step(rho_s)

    # 训练结束：直接输出训练过程中记录的最佳epoch与指标（不再进行最终加载与评估）
    if is_cross_dataset:
        fold_info = f" ({cross_source} -> {cross_target})"
    else:
        fold_info = f" (Fold {args.fold})" if hasattr(args, 'use_cv') and args.use_cv and hasattr(args, 'fold') else ""
    log_and_print(base_logger, "训练结束（跳过最终验证评估）")
    log_and_print(base_logger, f' EPOCH_best: {epoch_srcc_best}, SRCC_best: {rho_s_best:.6f}, PLCC_best: {rho_p_best:.6f}, KRCC_best: {rho_k_best:.6f}{fold_info}')

    timestamp = datetime.now().isoformat(timespec='seconds')
    common_header = [
        'dataset', 'ablation_mode', 'fold', 'SRCC', 'PLCC', 'KRCC', 'best_epoch',
        'run_id', 'gpu', 'seed', 'lr', 'backbone', 'text_encoder', 'log_info',
        'quality_preset', 'scheduler', 'warmup_epochs', 'train_aug_policy', 'use_ema', 'ema_decay',
        'loss_smooth_l1_weight', 'loss_pearson_weight', 'loss_rank_weight',
        'rank_min_diff', 'rank_temperature', 'rank_loss_type', 'label_norm',
        'label_norm_mean', 'label_norm_std', 'freeze_text_encoder',
        'train_SRCC', 'train_PLCC', 'train_KRCC', 'generalization_gap_srcc',
        'use_balanced_sampling', 'num_quality_bins', 'cv_protocol', 'timestamp'
    ]
    common_row = {
        'dataset': args.benchmark,
        'ablation_mode': getattr(args, 'ablation_mode', 'full'),
        'fold': 0 if (is_cross_dataset or is_official_split) else getattr(args, 'fold', 0),
        'SRCC': f'{rho_s_best:.6f}',
        'PLCC': f'{rho_p_best:.6f}',
        'KRCC': f'{rho_k_best:.6f}',
        'best_epoch': epoch_srcc_best,
        'run_id': getattr(args, 'run_id', ''),
        'gpu': getattr(args, 'gpu', ''),
        'seed': getattr(args, 'seed', ''),
        'lr': getattr(args, 'lr', ''),
        'backbone': getattr(args, 'backbone', ''),
        'text_encoder': getattr(args, 'text_encoder', ''),
        'log_info': getattr(args, 'log_info', ''),
        'quality_preset': getattr(args, 'quality_preset', 'none'),
        'scheduler': getattr(args, 'scheduler', ''),
        'warmup_epochs': getattr(args, 'warmup_epochs', ''),
        'train_aug_policy': getattr(args, 'train_aug_policy', ''),
        'use_ema': getattr(args, 'use_ema', False),
        'ema_decay': getattr(args, 'ema_decay', ''),
        'loss_smooth_l1_weight': getattr(args, 'loss_smooth_l1_weight', ''),
        'loss_pearson_weight': getattr(args, 'loss_pearson_weight', ''),
        'loss_rank_weight': getattr(args, 'loss_rank_weight', ''),
        'rank_min_diff': getattr(args, 'rank_min_diff', ''),
        'rank_temperature': getattr(args, 'rank_temperature', ''),
        'rank_loss_type': getattr(args, 'rank_loss_type', ''),
        'label_norm': label_norm_info['type'],
        'label_norm_mean': f"{label_norm_info['mean']:.6f}",
        'label_norm_std': f"{label_norm_info['std']:.6f}",
        'freeze_text_encoder': getattr(args, 'freeze_text_encoder', False),
        'train_SRCC': f"{best_train_metrics['srcc']:.6f}",
        'train_PLCC': f"{best_train_metrics['plcc']:.6f}",
        'train_KRCC': f"{best_train_metrics['krcc']:.6f}",
        'generalization_gap_srcc': f"{best_train_metrics['srcc'] - rho_s_best:.6f}",
        'use_balanced_sampling': getattr(args, 'use_balanced_sampling', False),
        'num_quality_bins': getattr(args, 'num_quality_bins', ''),
        'cv_protocol': 'cross-dataset-full-test-folds' if is_cross_dataset else (
            'official-train-val-test' if is_official_split else '5-fold-cv_folds'
        ),
        'timestamp': timestamp
    }

    if is_cross_dataset:
        csv_path = getattr(args, 'cross_results_csv', 'cross_dataset_results.csv')
        cross_header = [
            'source_dataset', 'target_dataset', 'dataset_pair',
            *common_header
        ]
        cross_row = {
            'source_dataset': cross_source,
            'target_dataset': cross_target,
            'dataset_pair': f'{cross_source}_to_{cross_target}',
            **common_row
        }
        append_locked_csv(csv_path, cross_header, cross_row)
        log_and_print(base_logger, f'📊 跨库结果已写入CSV: {csv_path}')
    elif hasattr(args, 'results_csv') and args.results_csv:
        append_locked_csv(args.results_csv, common_header, common_row)
        log_and_print(base_logger, f'📊 结果已写入CSV: {args.results_csv}')
