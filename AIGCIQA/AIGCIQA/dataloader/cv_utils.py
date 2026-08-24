"""
交叉验证数据加载器通用工具模块
提供交叉验证数据加载的公共功能和基类
"""

import torch
import torchvision.transforms as tr
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import numpy as np
import random

# 轻量获取CLIP tokenizer，避免为数据集初始化整套文本编码器
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("警告: 未找到open_clip，CLIP tokenizer不可用")

# 传统tokenizer作为备选
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class CrossValidationDataset(Dataset):
    """交叉验证数据集基类"""
    
    def __init__(self, image, label, text_prompt, transforms, text_encoder_path, image_names=None):
        self.image = image
        self.label = label
        self.text_prompt = text_prompt
        self.transforms = transforms
        self.image_names = image_names
        
        # 根据text_encoder_path选择tokenizer
        if text_encoder_path == "clip_convnext":
            if OPEN_CLIP_AVAILABLE:
                print("使用CLIP-ConvNeXt tokenizer进行tokenization")
                self.tokenizer = open_clip.get_tokenizer('convnext_large_d_320')
                self.use_clip_tokenizer = True
                self.max_length = 77
            elif TRANSFORMERS_AVAILABLE:
                print("警告: open_clip不可用，回退到transformer tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
                self.use_clip_tokenizer = False
                self.max_length = 512
            else:
                raise ImportError("open_clip和transformers均不可用，无法创建CLIP tokenizer")
        else:
            print(f"使用传统transformer tokenizer: {text_encoder_path}")
            if TRANSFORMERS_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
            else:
                raise ImportError("transformers库不可用且CLIPConvNeXt不可用")
            self.use_clip_tokenizer = False
            self.max_length = 512
        
        # 验证数据完整性
        if len(image) != len(label) or len(image) != len(text_prompt):
            raise ValueError(f"数据维度不匹配: 图像={len(image)}, 标签={len(label)}, 提示词={len(text_prompt)}")
            
        print(f"📊 交叉验证数据集初始化: {len(image)} 张图像")
        
        # 分析标签分布
        if len(label) > 0:
            min_score = min(label)
            max_score = max(label)
            mean_score = sum(label) / len(label)
            print(f"   标签分布: 最小={min_score:.3f}, 最大={max_score:.3f}, 平均={mean_score:.3f}")

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        data = {}
        
        # 图像处理
        data['image'] = self.transforms(self.image[idx])
        
        # 标签处理
        data['MOS_score'] = self.label[idx]
        
        # 文本处理
        text_prompt = self.text_prompt[idx]
        if not text_prompt or pd.isna(text_prompt):
            text_prompt = "empty prompt"
        
        if self.use_clip_tokenizer:
            # 使用CLIP tokenizer
            # CLIP tokenizer直接返回tensor，形状为[77]
            encoded_tokens = self.tokenizer([text_prompt])  # 输入列表
            
            # 确保正确的维度
            if encoded_tokens.dim() == 2:
                encoded_tokens = encoded_tokens.squeeze(0)  # 移除批次维度
            
            # CLIP不使用attention mask，创建一个全1的mask
            attention_mask = torch.ones_like(encoded_tokens)
            
            data['prompt'] = encoded_tokens
            data['attention_mask'] = attention_mask
        else:
            # 使用传统tokenizer
            encoded = self.tokenizer.encode_plus(
                text_prompt,
                add_special_tokens=True,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )
            
            # 确保维度正确 - 移除批次维度
            data['prompt'] = encoded['input_ids'].squeeze(0)
            data['attention_mask'] = encoded['attention_mask'].squeeze(0)
        
        # 可选的图像名称
        if self.image_names is not None:
            data['image_name'] = self.image_names[idx]
            
        return data


def load_cv_data_from_csv(csv_path, image_base_path, image_col='Image', prompt_col='Prompt', 
                         score_col='MOS', required_cols=None):
    """
    从CSV文件加载交叉验证数据
    
    Args:
        csv_path: CSV文件路径
        image_base_path: 图像文件基础路径  
        image_col: 图像文件名列名
        prompt_col: 提示词列名
        score_col: 评分列名
        required_cols: 必需的列名列表
        
    Returns:
        tuple: (image_list, label_list, text_prompt_list, image_names_list)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        
    # 读取CSV数据
    data = pd.read_csv(csv_path)
    print(f"📁 从 {csv_path} 加载数据: {len(data)} 条记录")
    
    # 检查必需的列
    if required_cols:
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"CSV文件缺少必需的列: {missing_cols}")
    
    # 提取数据
    image_names = data[image_col].tolist()
    text_prompts = data[prompt_col].tolist()
    scores = data[score_col].tolist()
    
    # 加载图像
    image_list = []
    valid_indices = []
    
    for i, image_name in enumerate(image_names):
        image_path = os.path.join(image_base_path, str(image_name))
        
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                image_list.append(image)
                valid_indices.append(i)
            except Exception as e:
                print(f"⚠️  加载图像失败 {image_path}: {e}")
        else:
            print(f"⚠️  图像文件不存在: {image_path}")
    
    # 过滤有效数据
    valid_image_names = [image_names[i] for i in valid_indices]
    valid_text_prompts = [text_prompts[i] for i in valid_indices]
    valid_scores = [scores[i] for i in valid_indices]
    
    print(f"✅ 成功加载 {len(image_list)} 张图像 (跳过 {len(image_names) - len(image_list)} 张)")
    
    return image_list, valid_scores, valid_text_prompts, valid_image_names


def get_image_transforms(args):
    """获取图像变换"""
    if args.backbone == 'clipconvnext':
        clip_mean = (0.48145466, 0.4578275, 0.40821073)
        clip_std = (0.26862954, 0.26130258, 0.27577711)
        aug_policy = getattr(args, 'train_aug_policy', 'legacy')
        benchmark = getattr(args, 'benchmark', '')

        if aug_policy == 'correspondence_stable' or benchmark in ['AIGCIQA2023c', 'AGIQA3Kc', 'I2IQAc']:
            train_transforms = tr.Compose([
                tr.Resize(336),
                tr.CenterCrop(320),
                tr.RandomHorizontalFlip(),
                tr.ToTensor(),
                tr.Normalize(mean=clip_mean, std=clip_std)
            ])
        elif aug_policy == 'quality_stable':
            train_transforms = tr.Compose([
                tr.Resize(336),
                tr.RandomResizedCrop(320, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                tr.RandomHorizontalFlip(),
                tr.ToTensor(),
                tr.Normalize(mean=clip_mean, std=clip_std)
            ])
        else:
            train_transforms = tr.Compose([
                tr.Resize(320),
                tr.RandomCrop(320),
                tr.RandomHorizontalFlip(),
                tr.ToTensor(),
                tr.Normalize(mean=clip_mean, std=clip_std)
            ])

        print(f"🖼️ clipconvnext训练增强策略: {aug_policy}")
        test_transforms = tr.Compose([
            tr.Resize(320),
            tr.CenterCrop(320),
            tr.ToTensor(),
            tr.Normalize(mean=clip_mean, std=clip_std)
        ])
        return train_transforms, test_transforms

    if args.backbone == 'inceptionv4':
        resize_img_size = 320
        crop_img_size = 299
    else:
        resize_img_size = 256
        crop_img_size = 224

    train_transforms = tr.Compose([
        tr.Resize(resize_img_size),
        tr.RandomCrop(crop_img_size),
        tr.RandomHorizontalFlip(),
        tr.ToTensor(),
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = tr.Compose([
        tr.Resize(resize_img_size),
        tr.CenterCrop(crop_img_size),
        tr.ToTensor(),
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, test_transforms


def create_cv_dataloader(images, labels, prompts, transforms, text_encoder_path, 
                        batch_size, shuffle=True, drop_last=True, image_names=None):
    """创建交叉验证数据加载器"""
    dataset = CrossValidationDataset(
        images, labels, prompts, transforms, text_encoder_path, image_names
    )
    
    num_workers = min(4, os.cpu_count() or 0)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    
    return dataloader


def validate_fold_number(fold, max_fold=10, min_fold=1):
    """验证fold编号的有效性"""
    if not isinstance(fold, int):
        raise TypeError(f"fold必须是整数，当前类型: {type(fold)}")

    if fold < min_fold or fold > max_fold:
        raise ValueError(f"fold编号超出范围: {fold} (有效范围: {min_fold}-{max_fold})")

    return True


def get_cv_folds_csv_paths(fold, dataset_key):
    """获取新版5折cv_folds中的训练和验证CSV路径"""
    validate_fold_number(fold, max_fold=5, min_fold=1)

    dataset_map = {
        'AGIQA1K': ('1k_cqupt', 'agiqa_1k_cqupt'),
        'AGIQA3K': ('3k_cqupt', 'agiqa_3k_cqupt'),
        'AGIQA3Kc': ('3k_cqupt', 'agiqa_3k_cqupt'),
        'AIGCIQA2023': ('2023_cqupt', 'agiqa_2023_cqupt'),
        'I2IQA': ('i2iqa_annotation_cqupt', 'i2iqa_annotation_cqupt'),
    }
    if dataset_key not in dataset_map:
        raise ValueError(f"不支持的cv_folds数据集: {dataset_key}")

    subdir, prefix = dataset_map[dataset_key]
    fold_dir = f"fold_{fold - 1}"
    fold_path = os.path.join(os.path.dirname(__file__), "cv_folds", subdir, fold_dir)
    train_csv_path = os.path.join(fold_path, f"{prefix}_train.csv")
    val_csv_path = os.path.join(fold_path, f"{prefix}_test.csv")

    return train_csv_path, val_csv_path

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
